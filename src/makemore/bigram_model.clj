(ns makemore.bigram-model
  (:require [clojure.string :as str]
            [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [libpython-clj2.python.ffi :as py-ffi]
            [nextjournal.clerk :as clerk]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]))

#_(clerk/serve! 
  {:watch-paths ["src/makemore"]
   :browse? true})

(require-python 'torch)
(require-python 'builtins)
(def F (py.- torch/nn functional))

(def words 
  (-> (slurp "resources/names.txt")
      (str/split-lines)))

(take 10 words)

(count words)

(->> words (map count) (apply min))

(->> words (map count) (apply max))

;bigrams
(def bigram-transform
  (comp (map str/lower-case) 
        (map #(str \. % \.))
        (mapcat #(partition 2 1 %))))

(->> words
    (into [] bigram-transform)
    (frequencies)
    (sort-by #(- (second %)))
    (take 5))

(def a (torch/zeros [3 5] :dtype torch/int32))
(py. (py/get-item a [1 3]) __iadd__ 1)

(def N (torch/zeros [27 27] :dtype torch/int32))
(def characters (->> words (str/join "") set sort))
(def stoi  (into {} (-> (map-indexed (fn [i c] [c (inc i)]) characters) 
                        (conj [\. 0]))))
(def itos (->> stoi (map (fn [[k v]] [v k])) (into {})))

(defn populate-tensor! [N]
  (doseq [[r c] (into [] bigram-transform words)]
    (py. (py/get-item N [(stoi r) (stoi c)]) __iadd__ 1)))

(populate-tensor! N)
(py/get-item N [3 3])

(defn chart-data 
  [N] 
  (for [i (range 27) j (range 27) ]
    {:text (str (itos i) (itos j) " " (py. (py/get-item N [i j]) item))
     :x j
     :y i
     :color (py. (py/get-item N [i j]) item)}))

(let [a (torch/zeros [27 27] :dtype torch/int32)
       _ (populate-tensor! a)]
 (clerk/vl 
  {:nextjournal.clerk/width :full} 
  (-> 
    (hc/xform 
    ht/corr-heatmap
    {:DATA (chart-data a)
     :WIDTH 1600
     :HEIGHT 1600 
     :COLS [:x :y]
     :COL1 :x :XTYPE "quantitative"
     :COL2 :y :YTYPE "quantitative"
     :CORR :color 
     :TXT :text 
     :TTYPE "nominal"
     :TEST "datum['color'] > 4000"})
    (assoc-in [:layer 1 :encoding :text :field] :text)
    (assoc-in [:layer 1 :encoding :text :type] "nominal") )))

(py/get-item N [(builtins/slice 0 1) ])

(defn make-p
  [ix]
  (as-> (py/get-item N [ix]) $ 
           (py. $ __div__ (py. $ sum) )))

(def g (py. (torch/Generator) manual_seed 2147483647))
(def p (make-p 0))

(defn get-ix [p g]
  (py. (torch/multinomial p :num_samples 1 :replacement true :generator g) item))

(loop [i 50
       g (py. (torch/Generator) manual_seed 2147483647)
       ix (get-ix (make-p 0) g)
       current ""
       out []]
  (if (zero? i)
    out
    (if (zero? ix) 
      (recur (dec i) g (get-ix (make-p 0) g) "" (conj out current))
      (recur i g (get-ix (make-p ix) g) (str current (itos ix)) out))))

;;optimized
;;check broadcast optimization
(py.- (py.. N float (sum 1 :keepdim true)) shape)
;;add 1 for model smoothing so that you dont get the loss as inf for bigrams that dont occur.
(def P (as-> (py. N float) $
              (py. $ __add__ 1)
              (py. $ __div__ (py. $ sum 1 :keepdim true))))
(py. (py/get-item P [0]) sum)

(loop [i 50
       g (py. (torch/Generator) manual_seed 2147483647)
       ix (get-ix (py/get-item P [0]) g)
       current ""
       out []]
  (if (zero? i)
    out
    (if (zero? ix) 
      (recur (dec i) g (get-ix (py/get-item P [0]) g) "" (conj out current))
      (recur i g (get-ix (py/get-item P [ix]) g) (str current (itos ix)) out))))

;; - quality of the model 
;; - GOAL: Maximize liklihood of the data w.r.t model parameters (statistical modeling)
;; - Equivalent to maximizing the log likelyhood (because log is monotonic)
;; - Equivalent to minimizing the negative log liklihood (negative since you are trying to minimize loss -negative log will have a minimum of 0)
;; - Equivalent to minimizing the average negative log likelihood

;;- log (a * b * c) = log (a) + log (b) + log (c)
(defn quality
  [test-or-train-set] 
  (->> test-or-train-set 
       (into [] bigram-transform)
       (reduce (fn [[log-likelihood cnt] [ch1 ch2]]
                 (let [ix1 (stoi ch1)
                       ix2 (stoi ch2)
                       prob (py/get-item P [ix1 ix2])
                       logprob (py. (torch/log prob) item)]
                   (println (str ch1 ch2) logprob)
                   [(+ log-likelihood logprob) (inc cnt)])) [0.0 0])))

(defn normalized-log-likelyhood
  [[log-likelihood cnt]]
  (/ (- log-likelihood) cnt))

(def train-set-quality (quality words))
;;negative log liklihood
(- (first train-set-quality))
;;normailized log likelyhood
(normalized-log-likelyhood train-set-quality)

;;after smoothing there is a legit value here
(-> ["andrejq"] quality normalized-log-likelyhood)

;;Now the fun part do it with a NN
;;The neural network will adjust the weight for the next char so that the mle is high

(defn make-tensors
  [words]
  (let [tr (fn
             ([] [[] []] )
             ([result] result) 
             ([[xs ys] [ch1 ch2]] [(conj xs (stoi ch1)) (conj ys (stoi ch2))]))
        [xs ys] (transduce bigram-transform tr [[] []] words)]
    [(torch/tensor xs) (torch/tensor ys)]))

(let [[xs ys] (make-tensors [(first words)])
      g (py. (torch/Generator) manual_seed 2147483647) ;randomly initialize 27 neurons each neuron receives 27 inputs
      W (torch/randn [27 27] :generator g :requires_grad true)
      xenc (py.. F (one_hot xs :num_classes 27) float) ;input to one hot network
      logits (torch/mm xenc W) ;predict log counts
      counts (py. logits exp)  ;counts equivalent to N
      probs (py. counts __div__ (py. counts sum 1 :keepdims true)) ;probability for next char  the last two lines are calles a softmax
      loss (py.. (py/get-item probs [(torch/arange 5) ys]) log mean __neg__)] 
  (println (py. loss item))
  (println (py. (py/get-item probs [0]) sum))
  (println (py.- probs shape))
  (py/set-attr! W 'grad nil)
  (py. loss backward)
  (println (py.- W grad)))


(defn train 
  [words iterations gradient-step]
  (let [[xs ys] (make-tensors words)
        g (py. (torch/Generator) manual_seed 2147483647) ;randomly initialize 27 neurons each neuron receives 27 inputs
        W (torch/randn [27 27] :generator g :requires_grad true)
        xenc (py.. F (one_hot xs :num_classes 27) float)
        num (py. xs nelement)] 
    (loop [k iterations
           W W]
      (if (zero? k)
        W
        (let [logits (torch/mm xenc W)
              counts (py. logits exp)  
              probs (py. counts __div__ (py. counts sum 1 :keepdims true))
              loss (py.. (py/get-item probs [(torch/arange num) ys]) log mean __neg__)]
          (println (py. loss item))
          (py. loss backward)
          (py. (py.- W data) __iadd__ (py.. (py.- W grad) (__mul__ gradient-step) __neg__ (__add__ (py.. W (__pow__ 2) (__mul__ 0.01) mean)))) ;regularization for smoothing
          (py. (py.- W grad) zero_)
          (recur (dec k) W))))))

(train [(first words)] 10 0.1)
(def W (train words 100 50))

(loop [i 50
       g (py. (torch/Generator) manual_seed 2147483647)
       ix 0
       current ""
       out []]
  (let [xenc (py.. F (one_hot (torch/tensor [ix]) :num_classes 27) float)         
        logits (torch/mm xenc W)         
        counts (py. logits exp) 
        p (py. counts __div__ (py. counts sum 1 :keepdims true))
        ix (get-ix p g)]
    (if (zero? i)
      out
      (if (zero? ix) 
        (recur (dec i) g 0 "" (conj out current))
        (recur i g ix (str current (itos ix)) out)))))
