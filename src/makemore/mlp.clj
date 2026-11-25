(ns makemore.mlp
  (:require [clojure.string :as str]
            [clj-pytorch.functional :as t]
            [clj-pytorch.context :as ctx]
            [scicloj.clay.v2.api :as clay]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]
            [libpython-clj2.python :as py :refer [py.- py.]]))

#_(clay/make! {:source-path "src/makemore/mlp.clj" :inline-js-and-css false})

(def words
  (-> (slurp "resources/names.txt")
      (str/split-lines)))

(count words)
(take 8 words)

(def characters (->> words (str/join "") set sort))
(def stoi  (into (sorted-map) (-> (map-indexed (fn [i c] [c (inc i)]) characters)
                                  (conj [\. 0]))))
(def itos (->> stoi (map (fn [[k v]] [v k])) (into (sorted-map))))

(defn make-xs-ys
  [block-size word]
  (->> (map-indexed (fn [i c]
                      (cond
                        (pos? (- block-size i)) [(str (apply str (repeat (- block-size i) \.)) (subs word 0 i)) c]
                        (= (inc i) (count word)) [(subs word (- i block-size) i) c]
                        :else [(subs word (- i block-size) i) c])) word)
       vec
       ((fn [res] (if (> (count word) block-size)
                    (conj res [(subs word (- (count word) block-size)) \.])
                    (conj res [(str (apply str (repeat (- block-size (count word)) \.)) word) \.]))))))

(defn block-transform
  [block-size]
  (comp (map str/lower-case)
        (mapcat (partial make-xs-ys block-size))))

(defn make-tensors
  [block-size words]
  (let [tr (fn
             ([] [[] []])
             ([result] result)
             ([[xs ys] [chars ch2]]  [(conj xs (map #(stoi %) chars)) (conj ys (stoi ch2))]))
        [xs ys] (transduce (block-transform block-size) tr [[] []] words)]
    [(t/tensor xs) (t/tensor ys)]))

#_(def x-y (make-tensors 3 (take 5 words)))
(def x-y (make-tensors 3 words))
(def X (first x-y))
(def Y (second x-y))
(def block-size 3)
(def embedding-size 10)
(def hidden-layer-neurons 200)
(def vocab-size 27)

(def C (t/randn [vocab-size embedding-size]))
;(def W1 (t/randn [(* block-size embedding-size) hidden-layer-neurons]))
;;to avoid activations in the flat region of tanh we reduce the weights at initialization so we dont have dead neurons
;(def W1 (t/mul (t/randn [(* block-size embedding-size) hidden-layer-neurons]) 0.2))
(def W1 (t/mul (t/randn [(* block-size embedding-size) hidden-layer-neurons]) (/ (/ 5 3) (Math/sqrt (* embedding-size block-size))))) ;;kaiming-normal not some random 0.2 to change deviation to 1
;(def b1 (t/randn [hidden-layer-neurons]))
(def b1 (t/mul (t/randn [hidden-layer-neurons]) 0.01))
;(def W2 (t/randn [hidden-layer-neurons vocab-size]))
(def W2 (t/mul (t/randn [hidden-layer-neurons vocab-size]) 0.01))
;(def b2 (t/randn [vocab-size]))
(def b2 (t/mul (t/randn [vocab-size]) 0))

(let [emb (t/index C X)
      h (-> (t/view emb [-1 (* embedding-size block-size)])
            #_(t/unbind emb :dim 1)
            #_(t/cat :dim 1)
            (t/matmul  W1)
            (t/add b1)
            (t/tanh-f))
      logits (t/add (t/matmul h W2) b2)
      loss (t/cross-entropy-f logits Y)
      #_#_counts (t/exp logits)
      #_#_prob (t/div counts (t/sum counts 1 true))
      #_#_loss (-> prob (t/index (t/arange 32)) t/log t/mean t/neg) ;(t/neg (t/mean (t/log (t/gather prob 1 (t/unsqueeze Y 1)))))
      parameters [C W1 b1 W2 b2]] ;More efficient to do this instead since no intermediate tensors created and uses clustered operations and backward pass is efficient since it uses "fused kernel" operations to calculate the derivative using previously saved values. Numerically well behaved. For very large value exp for logits will be inf 
  (println (t/shape X) (t/dtype X) (t/shape Y) (t/dtype Y))
  #_(println X)
  #_(println Y)
  (println "emb shape" (t/shape emb))
  (println "X at 13,2" (t/tensor-get X 13 2))
  (println "emb at 13 2 = c at X 13 2" (t/select (t/select emb 0 13) 0 2) (t/select C 0 (t/tensor-get X 13 2)))
  #_(println (-> (t/unbind emb :dim 1) (t/cat :dim 1) t/shape))
  (println "shape h" (t/shape h))
  (println "shape logits" (t/shape logits))
  #_(println logits)
  #_(println "shape prob" (t/shape prob))
  #_(println (t/sum (t/select prob 0 0)))
  #_(println loss torch-loss)
  (println "loss" loss)
  (println "params" (apply + (map t/numel parameters))))

;forward and backward  pass
(def shuffled-words (shuffle words))

(def n1 (int (* 0.8 (count shuffled-words))))
(def n2 (int (* 0.9 (count shuffled-words))))

(def train-set (make-tensors block-size (subvec shuffled-words 0 n1)))
(mapv t/shape train-set)
(def dev-set (make-tensors block-size (subvec shuffled-words n1 n2)))
(mapv t/shape dev-set)
(def test-set (make-tensors block-size (subvec shuffled-words n2)))
(mapv t/shape test-set)

(println "train" (mapv t/shape train-set) "dev" (mapv t/shape dev-set) "test" (mapv t/shape test-set))

(def parameters [C W1 b1 W2 b2])
;;parameters
(apply + (map t/numel parameters))

(doseq [p parameters] (py/set-attr! p "requires_grad" true))
(def batch-size 32)
(def lossi (atom []))

;; If this still grows native memory, wrap each iteration in py/stack-resource-context so
;; the tensors are released at the end of the step instead of waiting for the JVM GC.
(doseq [c (range 200000)
        :let [[X Y] train-set
              ix (t/randint 0 (first (t/shape X)) [batch-size])
              emb (t/index C (t/index X ix))
              h (-> (t/view emb [-1 (* embedding-size block-size)])
                    (t/matmul  W1)
                    (t/add b1)
                    (t/tanh-f))
              logits (t/add (t/matmul h W2) b2)
              loss (t/cross-entropy-f logits (t/index Y ix))]] ;More efficient to do this instead since no intermediate tensors created and uses clustered operations and backward pass is efficient since it uses "fused kernel" operations to calculate the derivative using previously saved values. Numerically well behaved. For very large value exp for logits will be inf 
  (doseq [p parameters] (py/set-attr! p "grad" nil))
  (py. loss backward)
  (ctx/no-grad (doseq [p parameters] (t/add! p (t/mul (t/grad p) (if (< c 100000) -0.1 -0.01)))))
  (swap! lossi conj (let [l (t/item loss)]
                      {:step c :loss l :log-loss (/ (Math/log l) (Math/log 10))}))
  (when (zero? (mod c 1000))
    (println "train loss" (t/item loss))))

(:loss (last @lossi))

;; log10 of the minibatch loss per step, as in the notebook
(-> (tc/dataset @lossi)
    (plotly/base {:=x :step :=y :log-loss :=width 1200})
    (plotly/layer-line))

;; embedding dimensions 0 and 1 for each character
;; is valid only when the embedding size is 2
(-> (tc/dataset (for [i (range vocab-size)]
                  {:x (t/tensor-get C i 0) :y (t/tensor-get C i 1) :char (str (itos i))}))
    (plotly/base {:=x :x :=y :y :=text :char :=width 700 :=height 700})
    (plotly/layer-point {:=mark-size 14})
    (plotly/plot)
    (assoc-in [:data 0 :mode] "markers+text")
    (assoc-in [:data 0 :textposition] "middle center"))

;;devset - hyperparemeter tuning
;; validationset final loss result
(defn validation-test
  [dev-or-test-set]
  (ctx/no-grad
   (let [[X Y] dev-or-test-set
         emb (t/index C X)
         h (-> (t/view emb [-1 (* embedding-size block-size)])
               (t/matmul  W1)
               (t/add b1)
               (t/tanh-f))
         logits (t/add (t/matmul h W2) b2)
         loss (t/cross-entropy-f logits Y)]
     (println "dev/test set loss" loss)
     loss)))
;;validation-loss
(validation-test validation-test)
;;test-loss
(validation-test test-set)

;;generate names

(defn sample-name
  []
  (reduce
   (fn [{:keys [context out]} _]
     (let [emb (t/index C (t/tensor [context]))
           h (-> (t/view emb [1 -1]) (t/matmul W1) (t/add b1) (t/tanh-f))
           probs (t/softmax (t/add (t/matmul h W2) b2) :dim 1)
           ix (t/item (t/multinomial probs 1))
           out (conj out ix)]
       (if (zero? ix)
         (reduced out)
         {:context (conj (subvec context 1) ix) :out out})))
   {:context (vec (repeat block-size 0)) :out []}
   (range)))

(ctx/no-grad
 (reduce (fn [names _] (conj names (apply str (map itos (sample-name)))))
         []
         (range 20)))
