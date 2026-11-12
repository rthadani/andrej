(ns makemore.bigram-model
  (:require [clojure.string :as str]
            [libpython-clj2.python :refer [py. py.-] :as py]
            [clj-pytorch.functional :as t]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.nn :refer [defmodule] :as nn]
            [clj-pytorch.optimizer :as optim]
            [nextjournal.clerk :as clerk]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]))

#_(clerk/serve! 
  {:watch-paths ["src/makemore"]
   :browse? true})

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

(def characters (->> words (str/join "") set sort))
(def stoi  (into {} (-> (map-indexed (fn [i c] [c (inc i)]) characters) 
                        (conj [\. 0]))))
(def itos (->> stoi (map (fn [[k v]] [v k])) (into {})))

;; no in-place indexed add in clj-pytorch, so count in clojure and build the tensor once
(defn make-counts []
  (let [freqs (frequencies (into [] bigram-transform words))]
    (t/tensor (vec (for [i (range 27)]
                     (vec (for [j (range 27)] (get freqs [(itos i) (itos j)] 0)))))
              :dtype t/int64)))

(def N (make-counts))
(t/tensor-get N 3 3)

(defn chart-data 
  [N] 
  (for [i (range 27) j (range 27) ]
    {:text (str (itos i) (itos j) " " (t/tensor-get N i j))
     :x j
     :y i
     :color (t/tensor-get N i j)}))

(let [a N]
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

(t/slice N [[0 1]])

(defn make-p
  [ix]
  (let [row (t/select N 0 ix)]
    (t/div row (t/sum row))))

;; clj-pytorch has no torch.Generator, so seed the global generator instead
(defn get-ix [p]
  (t/item (t/multinomial p 1 :replacement true)))

(t/manual-seed 2147483647)
(loop [i 50
       ix (get-ix (make-p 0))
       current ""
       out []]
  (if (zero? i)
    out
    (if (zero? ix) 
      (recur (dec i) (get-ix (make-p 0)) "" (conj out current))
      (recur i (get-ix (make-p ix)) (str current (itos ix)) out))))

;;optimized
;;check broadcast optimization
(t/shape (t/sum (tensor/->float N) 1 true))
;;add 1 for model smoothing so that you dont get the loss as inf for bigrams that dont occur.
(def P (let [smoothed (t/add (tensor/->float N) 1)]
         (t/div smoothed (t/sum smoothed 1 true))))
(t/sum (t/select P 0 0))

(t/manual-seed 2147483647)
(loop [i 50
       ix (get-ix (t/select P 0 0))
       current ""
       out []]
  (if (zero? i)
    out
    (if (zero? ix) 
      (recur (dec i) (get-ix (t/select P 0 0)) "" (conj out current))
      (recur i (get-ix (t/select P 0 ix)) (str current (itos ix)) out))))

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
                       logprob (t/item (t/log (t/select (t/select P 0 ix1) 0 ix2)))]
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

; xenc * W for emma 
; (5,27) mm (27, 27) -> (5,27)

; The  27 outputs are positive and negative numbers but they should be probabilities for the next character in the sequence
; - Need to interpret the 27 numbers so they represent probabilities.
; We need like what we had in the original table(see heatmap) each row had counts we normalized them to probabilities and the
; probabilities normalized to one. They cant be counts because counts are integers 
; the 27 outputs are logs to get the counts we need to exponentiate them e^x

(defn make-tensors
  [words]
  (let [tr (fn
             ([] [[] []] )
             ([result] result) 
             ([[xs ys] [ch1 ch2]] [(conj xs (stoi ch1)) (conj ys (stoi ch2))]))
        [xs ys] (transduce bigram-transform tr [[] []] words)]
    [(t/tensor xs) (t/tensor ys)]))

;; no one_hot in clj-pytorch
(defn one-hot [xs n]
  (tensor/->float (t/eq (t/unsqueeze xs 1) (t/arange n))))

(defn neg-log-likelihood [probs ys]
  (t/neg (t/mean (t/log (t/gather probs 1 (t/unsqueeze ys 1))))))

(let [[xs ys] (make-tensors [(first words)])
      _ (t/manual-seed 2147483647)
      W (tensor/requires-grad! (t/randn [27 27]) true) ;randomly initialize 27 neurons each neuron receives 27 inputs
      xenc (one-hot xs 27) ;input to one hot network
      logits (t/matmul xenc W) ;predict log counts
      counts (t/exp logits)  ;counts equivalent to N
      probs (t/div counts (t/sum counts 1 true)) ;probability for next char  the last two lines are calles a softmax
      loss (neg-log-likelihood probs ys)] ; gather picks the prob of the actual next char for each row (the arange indexing in the video)
  (println (t/item loss))
  (println (t/sum (t/select probs 0 0)))
  (println (t/shape probs))
  (py/set-attr! W 'grad nil)
  (optim/backward! loss)
  (println (py.- W grad)))


(defmodule BigramNN []
  :init (fn [self] (nn/register-parameter! self :W (t/randn [27 27]))) ;randomly initialize 27 neurons each neuron receives 27 inputs
  :forward (fn [self xenc] (t/matmul xenc (py/get-attr self "W"))))

(defn weights [model] (py/get-attr (.py-module model) "W"))

;; wrapping W in a module lets the optimizer do the update and grad zeroing
;; the 0.01 * mean(W^2) term in the loss is the regularization for smoothing
(defn train 
  [words iterations gradient-step]
  (let [[xs ys] (make-tensors words)
        _ (t/manual-seed 2147483647)
        model (BigramNN)
        opt (optim/sgd model :lr gradient-step :momentum 0)
        xenc (one-hot xs 27)] 
    (dotimes [_ iterations]
      (let [logits (nn/call model xenc)
            counts (t/exp logits)  
            probs (t/div counts (t/sum counts 1 true))
            loss (t/add (neg-log-likelihood probs ys)
                        (t/mean (t/mul (t/pow (weights model) 2) 0.01)))]
        (println (t/item loss))
        (optim/zero-grad! opt)
        (optim/backward! loss)
        (optim/step! opt)))
    (weights model)))

(train [(first words)] 10 0.1)
(def W (train words 100 50))

(t/manual-seed 2147483647)
(loop [i 50
       ix 0
       current ""
       out []]
  (let [xenc (one-hot (t/tensor [ix]) 27)
        logits (t/matmul xenc W)         
        counts (t/exp logits) 
        p (t/div counts (t/sum counts 1 true))
        ix (get-ix p)]
    (if (zero? i)
      out
      (if (zero? ix) 
        (recur (dec i) 0 "" (conj out current))
        (recur i ix (str current (itos ix)) out)))))
