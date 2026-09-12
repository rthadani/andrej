(ns gpt.gpt
  (:require [clojure.java.io :as io]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.functional :as t]
            [clj-pytorch.nn :refer [defmodule] :as nn]
            [clj-pytorch.optimizer :as optim]
            [clj-pytorch.context :as ctx]
            [libpython-clj2.python :as py :refer [py.-  py.]]
            [libpython-clj2.require :refer [require-python]]))

(def tinyshakespeare-url
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

(defn download-input
  ([]
   (download-input "resources/input.txt"))
  ([local-path]
   (let [file (io/file local-path)]
     (io/make-parents file)
     (println "Downloading from" tinyshakespeare-url "...")
     (with-open [in  (io/input-stream tinyshakespeare-url)
                 out (io/output-stream file)]
       (io/copy in out))
     (println "Saved to" (.getAbsolutePath file))
     (.getAbsolutePath file))))

#_(download-input)

(def text (slurp "resources/input.txt"))
(def chars (->> (set text) sort))
(def vocab-size (count chars))

(def stoi (into {} (map-indexed (fn [i c] [c i]) chars)))
(def itos (into {} (map-indexed (fn [i c] [i c]) chars)))

(defn encode
  [text]
  (mapv #(get stoi %) text))

(defn decode
  [v]
  (apply str (map #(get itos %) v)))

(def data (tensor/->tensor (encode text) :dtype t/long))
(def n (int (* 0.9 (count text))))

(def train-data (t/slice data [[0 n]]))
(def val-data (t/slice data [[n (count text)]]))

;;hyperparamters
(def block-size 256)
(def batch-size 32)
(def n-embed 384)
(def n-head 6)
(def n-layers 6)
(def max-iters 5000)
(def eval-interval 500)
(def learning-rate 3e-4)
(def eval-iters 200)
(def device (tensor/best-device))
(def dropout 0.2)

(t/manual-seed 1337)

(defn get-batch
  [split]
  (let [data (if (= (name split) "train") train-data val-data)
        max-start (- (t/numel data) block-size)
        ix (t/randint (- max-start block-size) [batch-size])
        ix-seq (tensor/->clj ix)
        x (->> ix-seq (map #(t/slice data [[% (+ % block-size)]])) (t/stack))
        y (->> ix-seq (map #(t/slice data [[(inc %) (+ % block-size 1)]])) (t/stack))]
    [(t/to-device x device) (t/to-device y device)]))

(defn estimate-loss
  [model eval-iters]
  (ctx/no-grad
   (nn/eval! model)
   (let [result (into {}
                      (for [split ["train" "val"]]
                        (let [mean-loss (->> (range eval-iters)
                                             (map (fn [_]
                                                    (let [[xb yb] (get-batch split)
                                                          [_ loss] (model xb yb)]
                                                      (t/item loss))))
                                             (reduce +)
                                             (* (/ 1.0 eval-iters)))]
                          [split mean-loss])))]
     (nn/train! model)
     result)))

(defmodule Head
  [head-size]
  :init (fn [self]
          (nn/register-buffer! self "tril" (t/tril (t/ones [block-size block-size] :device device))))
  :layers {:key (nn/linear n-embed head-size :bias false :device device)
           :query (nn/linear n-embed head-size :bias false :device device)
           :value (nn/linear n-embed head-size :bias false :device device)
           :dropout (nn/dropout dropout)}
  :forward (fn [self x]
             (let [[B T C] (t/shape x)
                   k ((nn/get-layer self :key) x)
                   q ((nn/get-layer self :query) x)
                   v ((nn/get-layer self :value) x)
                   wei (-> (t/matmul q (t/transpose k -2 -1))
                           (t/mul (/ 1.0 (Math/sqrt head-size)))
                           (t/masked-fill (t/eq (t/slice (py.- self tril) [[0 T] [0 T]]) 0) Double/NEGATIVE_INFINITY)
                           (t/softmax -1)
                           ((nn/get-layer self :dropout)))]
               (t/matmul wei v))))

(defmodule MultiHeadAttention
  [num-heads head-size]
  :layers {:heads (nn/module-list (for [_ (range num-heads)] (Head head-size)))
           :proj (nn/linear (* num-heads head-size) n-embed :device device)
           :dropout (nn/dropout dropout)}
  :forward (fn [self x]
             (-> (t/cat (mapv #(% x) (nn/module-list-seq self :heads)) :dim -1)
                 ((nn/get-layer self :proj))
                 ((nn/get-layer self :dropout)))))

(defmodule FeedForward
  [n-embed]
  :layers {:net (nn/sequential (nn/linear n-embed (* 4 n-embed) :device device) ;(* 4 n-embed)
                               (nn/relu)
                               (nn/linear (* 4 n-embed) n-embed :device device) ;projection
                               (nn/dropout dropout))}
  :forward (fn [self x]
             ((nn/get-layer self :net) x)))

(defmodule Block
  [n-embed n-head]
  :layers {:sa (MultiHeadAttention n-head (/ n-embed n-head))
           :ffwd (FeedForward n-embed)
           :ln1 (nn/layer-norm [n-embed] :device device)
           :ln2 (nn/layer-norm [n-embed] :device device)}
  :forward (fn [self x]
             (let [x (t/add x ((nn/get-layer self :sa) ((nn/get-layer self :ln1) x)))
                   x (t/add x ((nn/get-layer self :ffwd) ((nn/get-layer self :ln2) x)))]
               x)
             #_(-> x
                   ((nn/get-layer self :sa))
                   ((nn/get-layer self :ffwd)))))

(defmodule BigramLanguageModel
  []
  :layers {:token-embedding-table (nn/embedding vocab-size n-embed :device device)
           :position-embedding-table (nn/embedding block-size n-embed :device device)
           :blocks (apply nn/sequential (for [_ (range n-layers)] (Block n-embed n-head)))
           :ln-f (nn/layer-norm [n-embed] :device device)
           :lm-head (nn/linear n-embed vocab-size :device device)}
  :forward (fn [self idx & [targets]]
             (let [[B T] (t/shape idx)
                   token-emb ((nn/get-layer self :token-embedding-table) idx)
                   pos-emb ((nn/get-layer self :position-embedding-table) (t/arange T :device device))
                   x (t/add token-emb pos-emb)
                   logits (-> x
                              ((nn/get-layer self :blocks))
                              ((nn/get-layer self :ln-f))
                              ((nn/get-layer self :lm-head)))]
               (if (nil? targets)
                 [logits nil]
                 (let [[B T C] (t/shape logits)
                       logits  (t/view logits [(* B T) C])
                       targets (t/view targets [(* B T)])
                       loss    (when targets
                                 ((nn/cross-entropy-loss) logits targets))]
                   [logits loss])))))

(defn train
  [m]
  (let [optimizer (optim/adamw m :lr learning-rate)]
    (doseq [step (range max-iters)]
      (when (zero? (mod step eval-interval))
        (prn (estimate-loss m eval-iters)))
      (let [[xb yb]   (get-batch "train")
            [logits loss] (m xb yb)]
        (optim/zero-grad! optimizer)
        (optim/backward! loss)
        (optim/step! optimizer)))))

(defn generate
  [m idx max-new-tokens]
  (ctx/no-grad
   (reduce
    (fn [idx _]
      (let [[_ T]    (t/shape idx)
            start    (max 0 (- T block-size))
            idx-cond (t/slice idx [[nil nil] [start T]])
            [logits _] (m idx-cond)
            logits   (t/select logits 1 -1)   ; logits[:, -1, :] (B,T,C) -> (B,C), last timestep
            probs    (t/softmax logits)
            idx-next (t/multinomial probs 1)]
        (t/cat [idx idx-next] :dim 1)))
    idx
    (range max-new-tokens))))

(comment

  (def m (BigramLanguageModel))
  (train m)
  (def context (t/zeros [1 1] :dtype t/long :device device))
  (decode (-> m
              (generate context 500)
              (t/select 0 0) ;generate[0]
              tensor/->clj))

  ;;everything below this was the intermediate steps
  (encode "hii there")
  (decode (encode "hii there"))

  (def x (t/slice train-data [[0 block-size]]))
  (apply str (decode (tensor/->clj x)))
  (def y (t/slice train-data [[1 (inc block-size)]]))
  (doseq [t (range block-size)
          :let [context (t/slice x [[0 (inc t)]])
                target (t/tensor-get y t)]]
    (println (str "when input is " context " the target is " target)))

  (let [[xb yb] (get-batch "train")]
    (prn "inputs" (t/shape xb) xb)
    (prn "outputs " (t/shape yb) yb))

  (defmodule BigramLanguageModel
    []
    :layers {:token-embedding-table (nn/embedding vocab-size n-embed :device device)
             :lm-head (nn/linear n-embed vocab-size :device device)
             :position-embedding-table (nn/embedding n-embed vocab-size :device device)}
    :forward (fn [self idx & [targets]]
               (let [[B T] (t/shape idx)
                     token-emb ((nn/get-layer self :token-embedding-table) idx)
                     pos-emb ((nn/get-layer self :position-embedding-table) (t/arange T :device device))
                     x (t/add token-emb pos-emb)
                     logits ((nn/get-layer lm-head x))]
                 (if (nil? targets)
                   [logits nil]
                   (let [[B T C] (t/shape logits)
                         logits  (t/view logits [(* B T) C])
                         targets (t/view targets [(* B T)])
                         loss    (when targets
                                   ((nn/cross-entropy-loss) logits targets))]
                     [logits loss])))))

  (def m (BigramLanguageModel vocab-size))
  (let [[xb yb] (get-batch "train")
        [logits loss] (m xb yb)]
    (print (t/shape logits) loss)) ;loss = -ln(1/65)

  (defn generate
    [m idx max-new-tokens]
    (reduce
     (fn [idx _]
       (let [[logits _] (m idx)
             logits   (t/select logits 1 -1)   ; logits[:, -1, :] (B,T,C) -> (B,C), last timestep
             probs    (t/softmax logits)
             idx-next (t/multinomial probs 1)]
         (t/cat [idx idx-next] :dim 1)))
     idx
     (range max-new-tokens)))

  (decode (-> m
              (generate  (t/zeros [1 1] :dtype t/long :device device) 400)
              (t/select 0 0) ;generate[0]
              tensor/->clj))

  (def optimizer (optim/adamw m :lr 1e-3))
  ;;dont average losses while training
  (doseq [_ (range 1000)
          :let [[xb yb] (get-batch "train")
                [logits loss] (m xb yb)]]
    (optim/zero-grad! optimizer)
    (optim/backward! loss)
    (optim/step! optimizer)
    (prn (t/item loss)))

;;use the generate loss 
  (doseq [step (range 1000)]
    (when (zero? (mod step 100))
      (prn (estimate-loss m 200)))
    (let [[xb yb]       (get-batch "train")
          [logits loss] (m xb yb)]
      (optim/zero-grad! optimizer)
      (optim/backward! loss)
      (optim/step! optimizer)))

;;rerun decode here after training
;;
;;Mathematical trick in self attention
  (def B 4)
  (def T 8)
  (def C 32)
  (def x (t/randn [B T C]))
  (prn (t/shape x))
  (def xbow (t/zeros [B T C]))
  (doseq [b (range B)
          t (range T)
          :let [xprev (t/slice (t/select x 0 b) [[0 (inc t)]])]]
    (py/set-item! xbow [b t] (t/mean xprev 0)))

  ;;vectorize above
  (let [wei (as-> (t/tril (t/ones [T T])) $
              (t/div $ (t/sum $ 1 true)))
        xbow2 (t/matmul wei x)]
    (t/allclose xbow xbow2))

  ;;with softmax
  (let [tril  (t/tril (t/ones [T T]))
        wei   (-> (t/zeros [T T])
                  (t/masked-fill (t/eq tril 0) Double/NEGATIVE_INFINITY)
                  (t/softmax -1))
        xbow3 (t/matmul wei x)]
    (t/allclose xbow xbow3))

;;single head perform self attention
  (def head-size 16)
  (def key (nn/linear C head-size false))
  (def query (nn/linear C head-size false))
  (def k (key x)) ;(B T 16)
  (def q (query x)) ;(B T:wei 16)
  (def v (value x)) ;(B T:wei 16)
  (def wei (-> (t/matmul q (t/transpose k -2 -1))
               (t/masked-fill (t/eq (t/tril (t/ones [T T])) 0) Double/NEGATIVE_INFINITY)
               (t/softmax -1)))
  (def out (t/matmul wei v)))

