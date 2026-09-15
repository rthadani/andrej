(ns gpt.nanogpt
  "nanochat GPT: RoPE, QK-norm, ReLU2, GQA, value embeddings, smear, backout.
   Uses SDPA in place of Flash Attention 3. No KV-cache or sliding window."
  (:require [clj-pytorch.tensor    :as tensor]
            [clj-pytorch.functional :as t]
            [clj-pytorch.nn        :refer [defmodule] :as nn]
            [clj-pytorch.optimizer :as optim]
            [clj-pytorch.context   :as ctx]
            [libpython-clj2.python :as py :refer [py.- py.]]))

;;; ---------------------------------------------------------------------------
;;; Config

(defn make-config
  [& {:keys [sequence-len vocab-size n-layer n-head n-kv-head n-embd]
      :or   {sequence-len 2048
             vocab-size   32768
             n-layer      12
             n-head       6
             n-kv-head    6
             n-embd       768}}]
  {:sequence-len sequence-len
   :vocab-size   vocab-size
   :n-layer      n-layer
   :n-head       n-head
   :n-kv-head    n-kv-head
   :n-embd       n-embd})

(def device (tensor/best-device))

;;; ---------------------------------------------------------------------------
;;; Utilities

(defn norm
  "RMSNorm with no learnable parameters."
  [x]
  (t/rms-norm x [(last (t/shape x))]))

(defn has-ve?
  "True when layer layer-idx should have a value embedding (alternating, last included)."
  [layer-idx n-layer]
  (= (mod layer-idx 2) (mod (- n-layer 1) 2)))

(defn apply-rotary-emb
  "Rotate x of shape (B T H D) using RoPE cos/sin of shape (1 T 1 D/2)."
  [x cos sin]
  (let [d  (/ (nth (t/shape x) 3) 2)
        x1 (t/slice x [:all :all :all [0 d]])
        x2 (t/slice x [:all :all :all [d (* 2 d)]])
        y1 (t/add (t/mul x1 cos) (t/mul x2 sin))
        y2 (t/add (t/mul x1 (t/neg sin)) (t/mul x2 cos))]
    (t/cat [y1 y2] :dim 3)))

(defn precompute-rotary-emb
  "Precompute cos/sin tables of shape (1 seq-len 1 head-dim/2) for RoPE."
  [seq-len head-dim dev & {:keys [base] :or {base 100000}}]
  (let [ch  (t/arange 0 head-dim 2 :dtype t/float32 :device dev)
        inv (t/div 1.0 (t/scalar-pow base (t/div ch head-dim)))
        ts  (t/arange seq-len :dtype t/float32 :device dev)
        fq  (t/outer ts inv)
        cos (-> (t/cos fq) (t/unsqueeze 0) (t/unsqueeze 2))
        sin (-> (t/sin fq) (t/unsqueeze 0) (t/unsqueeze 2))]
    [cos sin]))

;;; ---------------------------------------------------------------------------
;;; CausalSelfAttention

(defmodule CausalSelfAttention
  [cfg layer-idx]
  :layers (let [{:keys [n-embd n-head n-kv-head n-layer]} cfg
                hd   (/ n-embd n-head)
                base {:c-q    (nn/linear n-embd (* n-head hd)    :bias false :device device)
                      :c-k    (nn/linear n-embd (* n-kv-head hd) :bias false :device device)
                      :c-v    (nn/linear n-embd (* n-kv-head hd) :bias false :device device)
                      :c-proj (nn/linear n-embd n-embd           :bias false :device device)}]
            (if (has-ve? layer-idx n-layer)
              (assoc base :ve-gate (nn/linear 12 n-kv-head :bias false :device device))
              base))
  :forward (fn [self x ve cos sin]
             (let [{:keys [n-embd n-head n-kv-head]} cfg
                   hd       (/ n-embd n-head)
                   [B T _]  (t/shape x)
                   q        (t/view ((nn/get-layer self :c-q) x) [B T n-head hd])
                   k        (t/view ((nn/get-layer self :c-k) x) [B T n-kv-head hd])
                   v        (t/view ((nn/get-layer self :c-v) x) [B T n-kv-head hd])
                   v        (if (some? ve)
                              (let [ve-r (t/view ve [B T n-kv-head hd])
                                    gate (t/mul (t/sigmoid-f
                                                 ((nn/get-layer self :ve-gate)
                                                  (t/slice x [:all :all [0 12]])))
                                                3.0)
                                    gate (t/unsqueeze gate -1)]
                                (t/add v (t/mul gate ve-r)))
                              v)
                   q        (t/mul (norm (apply-rotary-emb q cos sin)) 1.2)
                   k        (t/mul (norm (apply-rotary-emb k cos sin)) 1.2)
                   q        (t/transpose q 1 2)
                   k        (t/transpose k 1 2)
                   v        (t/transpose v 1 2)
                   grp      (/ n-head n-kv-head)
                   k        (if (> grp 1) (t/repeat-interleave k grp :dim 1) k)
                   v        (if (> grp 1) (t/repeat-interleave v grp :dim 1) v)
                   y        (t/scaled-dot-product-attention q k v :is-causal true)
                   y        (t/view (t/contiguous (t/transpose y 1 2)) [B T -1])]
               ((nn/get-layer self :c-proj) y))))

;;; ---------------------------------------------------------------------------
;;; MLP

(defmodule MLP [cfg]
  :layers {:c-fc   (nn/linear (:n-embd cfg) (* 4 (:n-embd cfg)) :bias false :device device)
           :c-proj (nn/linear (* 4 (:n-embd cfg)) (:n-embd cfg) :bias false :device device)}
  :forward (fn [self x]
             (-> x
                 ((nn/get-layer self :c-fc))
                 (t/relu-f)
                 (t/square)
                 ((nn/get-layer self :c-proj)))))

;;; ---------------------------------------------------------------------------
;;; Block

(defmodule Block [cfg layer-idx]
  :layers {:attn (CausalSelfAttention cfg layer-idx)
           :mlp  (MLP cfg)}
  :forward (fn [self x ve cos sin]
             (let [x (t/add x ((nn/get-layer self :attn) (norm x) ve cos sin))
                   x (t/add x ((nn/get-layer self :mlp)  (norm x)))]
               x)))

;;; ---------------------------------------------------------------------------
;;; GPT

(defmodule GPT [cfg]
  :layers (let [{:keys [vocab-size n-embd n-layer n-head n-kv-head]} cfg
                pad    64
                pv     (* (quot (+ vocab-size pad -1) pad) pad)
                hd     (/ n-embd n-head)
                kv-dim (* n-kv-head hd)
                base   {:wte        (nn/embedding pv n-embd :device device)
                        :blocks     (nn/module-list (for [i (range n-layer)] (Block cfg i)))
                        :smear-gate (nn/linear 24 1 :bias false :device device)
                        :lm-head    (nn/linear n-embd pv :bias false :device device)}]
            (reduce (fn [m i]
                      (if (has-ve? i n-layer)
                        (assoc m (keyword (str "ve-" i)) (nn/embedding pv kv-dim :device device))
                        m))
                    base
                    (range n-layer)))
  :init (fn [self]
          (let [{:keys [n-embd n-head sequence-len n-layer]} cfg
                hd        (/ n-embd n-head)
                [cos sin] (precompute-rotary-emb (* 10 sequence-len) hd device)]
            (nn/register-buffer! self "cos" cos)
            (nn/register-buffer! self "sin" sin)
            (nn/register-parameter! self "resid-lambdas"
                                    (t/ones [n-layer] :device device))
            (nn/register-parameter! self "x0-lambdas"
                                    (t/zeros [n-layer] :device device))
            (nn/register-parameter! self "smear-lambda"
                                    (t/zeros [1] :device device))
            (nn/register-parameter! self "backout-lambda"
                                    (t/mul (t/ones [1] :device device) 0.2))))
  :forward (fn [self idx & [targets]]
             (let [{:keys [n-embd n-head n-layer vocab-size sequence-len]} cfg
                   [B T]      (t/shape idx)
                   cos        (t/slice (py.- self cos) [:all [0 T] :all :all])
                   sin        (t/slice (py.- self sin) [:all [0 T] :all :all])
                   x          ((nn/get-layer self :wte) idx)
                   x          (norm x)
                   x          (if (> T 1)
                                (let [gate    (t/mul (py.- self smear_lambda)
                                                     (t/sigmoid-f ((nn/get-layer self :smear-gate)
                                                                   (t/slice x [:all [1 T] [0 24]]))))
                                      x-head  (t/slice x [:all [0 1] :all])
                                      x-tail  (t/slice x [:all [1 T] :all])
                                      x-prev  (t/slice x [:all [0 (- T 1)] :all])]
                                  (t/cat [x-head (t/add x-tail (t/mul gate x-prev))] :dim 1))
                                x)
                   x0         x
                   bo-idx     (quot n-layer 2)
                   blocks     (nn/module-list-seq self :blocks)
                   [x x-bo]   (reduce
                               (fn [[x x-bo] [i blk]]
                                 (let [rl   (py/get-item (py.- self resid_lambdas) i)
                                       x0l  (py/get-item (py.- self x0_lambdas) i)
                                       x    (t/add (t/mul x rl) (t/mul x0 x0l))
                                       ve   (when (has-ve? i n-layer)
                                              ((nn/get-layer self (keyword (str "ve-" i))) idx))
                                       x    (blk x ve cos sin)
                                       xbo  (if (= i bo-idx) x x-bo)]
                                   [x xbo]))
                               [x nil]
                               (map-indexed vector blocks))
                   x          (if x-bo
                                (t/sub x (t/mul (py.- self backout_lambda) x-bo))
                                x)
                   x          (norm x)
                   softcap    15.0
                   logits     ((nn/get-layer self :lm-head) x)
                   logits     (t/slice logits [:all :all [0 vocab-size]])
                   logits     (tensor/->float logits)
                   logits     (t/mul (t/tanh-f (t/div logits softcap)) softcap)]
               (if (some? targets)
                 (let [C    (last (t/shape logits))
                       loss (t/cross-entropy-f (t/view logits [(* B T) C])
                                               (t/view targets [(* B T)])
                                               :ignore-index -1)]
                   [logits loss])
                 [logits nil]))))

;;; ---------------------------------------------------------------------------
;;; Training

(defn train
  [m max-iters get-batch
   & {:keys [learning-rate eval-interval eval-iters]
      :or   {learning-rate 3e-4 eval-interval 500 eval-iters 200}}]
  (let [optimizer (optim/adamw m :lr learning-rate)]
    (doseq [step (range max-iters)]
      (when (zero? (mod step eval-interval))
        (println "step" step))
      (let [[xb yb]    (get-batch)
            [_ loss]   (m xb yb)]
        (optim/zero-grad! optimizer)
        (optim/backward! loss)
        (optim/step! optimizer)))))

;;; ---------------------------------------------------------------------------
;;; Generation

(defn generate
  [m tokens max-tokens
   & {:keys [temperature top-k seq-len] :or {temperature 1.0 seq-len 2048}}]
  (ctx/no-grad
   (loop [ids (t/to-device (tensor/->tensor [(vec tokens)] :dtype t/long) device)
          out (vec tokens)]
     (if (>= (count out) (+ (count tokens) max-tokens))
       out
       (let [[_ T]    (t/shape ids)
             start    (max 0 (- T seq-len))
             ctx-ids  (t/slice ids [:all [start T]])
             result   (m ctx-ids)
             logits   (py/get-item result 0)
             logits   (t/select logits 1 -1)
             logits   (if top-k
                        (let [k      (min top-k (last (t/shape logits)))
                              vals   (:values (t/topk logits k))
                              thr    (t/unsqueeze (t/select vals 1 -1) 1)]
                          (t/masked-fill logits (t/lt logits thr) Double/NEGATIVE_INFINITY))
                        logits)
             logits   (t/div logits (double temperature))
             probs    (t/softmax logits)
             next-id  (t/multinomial probs 1)
             new-ids  (t/cat [ids next-id] :dim 1)
             tok      (int (t/item (t/select next-id 0 0)))]
         (recur new-ids (conj out tok)))))))

(comment
  ;; Quick smoke test
  (def cfg (make-config :n-layer 6 :n-head 6 :n-kv-head 6 :n-embd 384
                        :vocab-size 32768 :sequence-len 512))
  (def m (GPT cfg))
  (def idx (t/to-device (tensor/->tensor [[0 1 2 3]] :dtype t/long) device))
  (m idx)
  (nn/summary m [(:sequence-len cfg)] :batch-size 1 :device device :dtypes [t/long]))

(comment
  ;; Shakespeare training — reuse gpt.clj's data loading and tokeniser
  (require '[gpt.gpt :as gpt])

  (def block-size 256)
  (def batch-size 16)

  (def cfg (make-config
            :n-layer  6
            :n-head   6
            :n-kv-head 6
            :n-embd   384
            :vocab-size (count gpt/chars)
            :sequence-len block-size))

  (def m (GPT cfg))

  (defn get-batch []
    (let [data     gpt/train-data
          max-start (- (t/numel data) block-size)
          ix       (t/randint max-start [batch-size])
          ix-seq   (tensor/->clj ix)
          x (->> ix-seq (map #(t/slice data [[% (+ % block-size)]])) t/stack)
          y (->> ix-seq (map #(t/slice data [[(inc %) (+ % block-size 1)]])) t/stack)]
      [(t/to-device x device) (t/to-device y device)]))

;; Train
  (train m 5000 get-batch
         :learning-rate 3e-4
         :eval-interval 500)

  ;; Generate
  (let [context (t/to-device (tensor/->tensor [[0]] :dtype t/long) device)
        tokens  (generate m [0] 500)]
    (gpt/decode tokens)))
