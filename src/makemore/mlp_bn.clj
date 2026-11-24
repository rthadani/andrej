(ns makemore.mlp-bn
  "makemore part 3: batch normalization, first inline in the MLP and then as a
  stack of Linear, BatchNorm1d and Tanh layers."
  (:require [clojure.string :as str]
            [clj-pytorch.functional :as t]
            [clj-pytorch.tensor :as tensor]
            [clj-pytorch.context :as ctx]
            [clj-pytorch.optimizer :as optim]
            [scicloj.clay.v2.api :as clay]
            [scicloj.tableplot.v1.plotly :as plotly]
            [tablecloth.api :as tc]
            [libpython-clj2.python :as py]))

#_(clay/make! {:source-path "src/makemore/mlp_bn.clj" :inline-js-and-css false})

;; Data

(def words
  (-> (slurp "resources/names.txt")
      (str/split-lines)))

(def characters (->> words (str/join "") set sort))
(def stoi (into (sorted-map) (-> (map-indexed (fn [i c] [c (inc i)]) characters)
                                 (conj [\. 0]))))
(def itos (->> stoi (map (fn [[k v]] [v k])) (into (sorted-map))))
(def vocab-size (count itos))

(def block-size 3)

(defn word-rows
  "One [context next-ix] row per character of the word, plus the end token."
  [word]
  (let [ixs (concat (map stoi (str/lower-case word)) [0])]
    (first (reduce (fn [[rows context] ix]
                     [(conj rows [context ix]) (conj (subvec context 1) ix)])
                   [[] (vec (repeat block-size 0))]
                   ixs))))

(defn build-dataset
  [words]
  (let [rows (mapcat word-rows words)]
    [(t/tensor (mapv first rows)) (t/tensor (mapv second rows))]))

(def shuffled-words (shuffle words))
(def n1 (int (* 0.8 (count shuffled-words))))
(def n2 (int (* 0.9 (count shuffled-words))))

(def train-set (build-dataset (subvec shuffled-words 0 n1)))
(def dev-set (build-dataset (subvec shuffled-words n1 n2)))
(def test-set (build-dataset (subvec shuffled-words n2)))
(def splits {:train train-set :val dev-set :test test-set})

(println "train" (mapv t/shape train-set) "dev" (mapv t/shape dev-set) "test" (mapv t/shape test-set))

(def batch-size 32)

(defn minibatch
  [[X Y]]
  (let [ix (t/randint 0 (first (t/shape X)) [batch-size])]
    [(t/index X ix) (t/index Y ix)]))

(defn sgd-step!
  [parameters loss lr]
  (doseq [p parameters] (py/set-attr! p "grad" nil))
  (optim/backward! loss)
  (ctx/no-grad (doseq [p parameters] (t/add! p (t/mul (t/grad p) (- lr))))))

(defn log10 [x] (/ (Math/log x) (Math/log 10)))

(defn sample-name
  "Samples next characters from logits-fn until the end token. logits-fn takes a
  (1, block-size) index tensor and returns (1, vocab-size) logits."
  [logits-fn]
  (reduce
   (fn [{:keys [context out]} _]
     (let [probs (t/softmax (logits-fn (t/tensor [context])) :dim 1)
           ix (t/item (t/multinomial probs 1))
           out (conj out ix)]
       (if (zero? ix)
         (reduced out)
         {:context (conj (subvec context 1) ix) :out out})))
   {:context (vec (repeat block-size 0)) :out []}
   (range)))

(defn sample-names
  [logits-fn n]
  (ctx/no-grad
   (mapv (fn [_] (apply str (map itos (sample-name logits-fn)))) (range n))))

;; MLP with an inline batch norm layer

(def n-embd 10)
(def n-hidden 200)
(def max-steps 200000)

(t/manual-seed 2147483647)

(def C (t/randn [vocab-size n-embd]))
;; kaiming normal with the tanh gain, so the preactivations start roughly unit gaussian
(def W1 (t/mul (t/randn [(* n-embd block-size) n-hidden])
               (/ (/ 5 3) (Math/sqrt (* n-embd block-size)))))
;; no b1, the batch norm bias makes it redundant
(def W2 (t/mul (t/randn [n-hidden vocab-size]) 0.01))
(def b2 (t/mul (t/randn [vocab-size]) 0))

(def bngain (t/ones [1 n-hidden]))
(def bnbias (t/zeros [1 n-hidden]))
(def bn-running (atom {:mean (t/zeros [1 n-hidden]) :std (t/ones [1 n-hidden])}))

(def parameters [C W1 W2 b2 bngain bnbias])
(println "parameters" (apply + (map t/numel parameters)))
(doseq [p parameters] (tensor/requires-grad! p true))

(defn hidden-preact
  [Xb]
  (-> (t/index C Xb)
      (t/view [-1 (* n-embd block-size)])
      (t/matmul W1)))

(defn logits-from-preact
  [hpreact]
  (-> hpreact (t/tanh-f) (t/matmul W2) (t/add b2)))

(defn batch-norm
  [hpreact mean std]
  (t/add (t/mul bngain (t/div (t/sub hpreact mean) std)) bnbias))

(defn train-step!
  [[Xb Yb] lr]
  (let [hpreact (hidden-preact Xb)
        bnmeani (t/mean hpreact 0 true)
        bnstdi (t/std hpreact 0 true)
        loss (t/cross-entropy-f
              (logits-from-preact (batch-norm hpreact bnmeani bnstdi))
              Yb)]
    (ctx/no-grad
     (swap! bn-running (fn [{:keys [mean std]}]
                         {:mean (t/add (t/mul mean 0.999) (t/mul bnmeani 0.001))
                          :std (t/add (t/mul std 0.999) (t/mul bnstdi 0.001))})))
    (sgd-step! parameters loss lr)
    (t/item loss)))

(def lossi (atom []))

(dotimes [i max-steps]
  (let [lr (if (< i 100000) 0.1 0.01)
        l (train-step! (minibatch train-set) lr)]
    (when (zero? (mod i 10000))
      (println (format "%7d/%7d: %.4f" i max-steps l)))
    (swap! lossi conj (log10 l))))

;; mean log10 loss per 1000 steps, the raw series is too noisy to read
(-> (tc/dataset (map-indexed (fn [i chunk] {:step (* i 1000) :log-loss (/ (reduce + chunk) (count chunk))})
                             (partition-all 1000 @lossi)))
    (plotly/base {:=x :step :=y :log-loss :=width 1200})
    (plotly/layer-line))

;; Fix the batch norm statistics by passing the whole training set through once.
;; The running averages from training should land in the same place.
(def bn-calibrated
  (ctx/no-grad
   (let [hpreact (hidden-preact (first train-set))]
     {:mean (t/mean hpreact 0 true) :std (t/std hpreact 0 true)})))

(defn split-loss
  ([split] (split-loss split @bn-running))
  ([split {:keys [mean std]}]
   (ctx/no-grad
    (let [[X Y] (splits split)
          logits (logits-from-preact (batch-norm (hidden-preact X) mean std))
          loss (t/item (t/cross-entropy-f logits Y))]
      (println split loss)
      loss))))

(split-loss :train)
(split-loss :val)
(split-loss :train bn-calibrated)
(split-loss :val bn-calibrated)

(sample-names (fn [Xb]
                (let [{:keys [mean std]} @bn-running]
                  (logits-from-preact (batch-norm (hidden-preact Xb) mean std))))
              20)

;; The same network as a stack of layers

(defmulti forward (fn [layer _x] (:kind layer)))
(defmulti layer-params :kind)

(defn linear
  [fan-in fan-out & {:keys [bias] :or {bias true}}]
  {:kind :linear
   :weight (t/div (t/randn [fan-in fan-out]) (Math/sqrt fan-in))
   :bias (when bias (t/zeros [fan-out]))
   :out (atom nil)})

(defmethod forward :linear
  [{:keys [weight bias out]} x]
  (reset! out (cond-> (t/matmul x weight)
                bias (t/add bias))))

(defmethod layer-params :linear
  [{:keys [weight bias]}]
  (cond-> [weight] bias (conj bias)))

(defn batchnorm1d
  [dim & {:keys [eps momentum] :or {eps 1e-5 momentum 0.1}}]
  {:kind :batchnorm
   :eps eps
   :momentum momentum
   :training (atom true)
   :gamma (t/ones [dim])
   :beta (t/zeros [dim])
   :running-mean (atom (t/zeros [dim]))
   :running-var (atom (t/ones [dim]))
   :out (atom nil)})

(defmethod forward :batchnorm
  [{:keys [eps momentum training gamma beta running-mean running-var out]} x]
  (let [training? @training
        xmean (if training? (t/mean x 0 true) @running-mean)
        xvar (if training? (t/variance x 0 true) @running-var)
        xhat (t/div (t/sub x xmean) (t/sqrt (t/add xvar eps)))]
    (when training?
      (ctx/no-grad
       (swap! running-mean #(t/add (t/mul % (- 1 momentum)) (t/mul xmean momentum)))
       (swap! running-var #(t/add (t/mul % (- 1 momentum)) (t/mul xvar momentum)))))
    (reset! out (t/add (t/mul gamma xhat) beta))))

(defmethod layer-params :batchnorm
  [{:keys [gamma beta]}]
  [gamma beta])

(defn tanh-layer
  []
  {:kind :tanh :out (atom nil)})

(defmethod forward :tanh
  [{:keys [out]} x]
  (reset! out (t/tanh-f x)))

(defmethod layer-params :tanh [_] [])

(defn set-training!
  [layers training?]
  (doseq [{:keys [kind training]} layers
          :when (= kind :batchnorm)]
    (reset! training training?)))

(defn forward-all
  [layers x]
  (reduce (fn [x layer] (forward layer x)) x layers))

(def deep-hidden 100)
(def deep-steps 1000) ;the notebook stops here to look at the statistics, a full run is 200000

(defn make-net
  []
  (let [hidden (mapcat (fn [i]
                         [(linear (if (zero? i) (* n-embd block-size) deep-hidden) deep-hidden :bias false)
                          (batchnorm1d deep-hidden)
                          (tanh-layer)])
                       (range 5))
        layers (vec (concat hidden [(linear deep-hidden vocab-size :bias false)
                                    (batchnorm1d vocab-size)]))]
    {:C (t/randn [vocab-size n-embd])
     ;; last layer is made less confident
     :layers (update-in layers [(dec (count layers)) :gamma] #(t/mul % 0.1))}))

(defn net-params
  [{:keys [C layers]}]
  (into [C] (mapcat layer-params) layers))

(defn net-logits
  [{:keys [C layers]} Xb]
  (forward-all layers (-> (t/index C Xb) (t/view [-1 (* n-embd block-size)]))))

(defn train-net!
  "Trains for steps minibatches and returns the per step log10 loss and the per step,
  per parameter log10 of the update to data ratio."
  [net steps]
  (let [params (net-params net)]
    (doseq [p params] (tensor/requires-grad! p true))
    (reduce
     (fn [{:keys [lossi ud]} i]
       (let [[Xb Yb] (minibatch train-set)
             lr (if (< i 150000) 0.1 0.01)
             loss (t/cross-entropy-f (net-logits net Xb) Yb)]
         ;; the layer outputs are not leaves, so ask for their gradients to be kept for the plots
         (doseq [{:keys [out]} (:layers net)] (py/call-attr @out "retain_grad"))
         (sgd-step! params loss lr)
         (when (zero? (mod i 100))
           (println (format "%7d/%7d: %.4f" i steps (t/item loss))))
         {:lossi (conj lossi (log10 (t/item loss)))
          :ud (conj ud (ctx/no-grad
                        (mapv #(log10 (t/item (t/div (t/std (t/mul (t/grad %) lr)) (t/std %)))) params)))}))
     {:lossi [] :ud []}
     (range steps))))

(t/manual-seed 2147483647)
(def net (make-net))
(println "parameters" (apply + (map t/numel (net-params net))))
(def stats (train-net! net deep-steps))

(defn hist-rows
  [t label]
  (let [{:keys [hist bin-edges]} (t/histogram (t/detach t) :density true)]
    (map (fn [x y] {:x x :y y :series label})
         (butlast (tensor/->clj bin-edges))
         (tensor/->clj hist))))

(defn hist-plot
  [rows]
  (-> (tc/dataset rows)
      (plotly/base {:=x :x :=y :y :=color :series :=width 1200})
      (plotly/layer-line)))

(defn tanh-layers
  "Index and layer pairs for the tanh layers, leaving out the output layer."
  [{:keys [layers]}]
  (filter (fn [[_ l]] (= :tanh (:kind l)))
          (map-indexed vector (butlast layers))))

;; activation distribution
(hist-plot
 (mapcat (fn [[i {:keys [out]}]]
           (let [x @out]
             (println (format "layer %d: mean %+.2f, std %.2f, saturated: %.2f%%" i
                              (t/item (t/mean x)) (t/item (t/std x))
                              (* 100 (t/item (t/mean (tensor/->float (t/gt (t/abs x) 0.97)))))))
             (hist-rows x (str "layer " i " (tanh)"))))
         (tanh-layers net)))

;; gradient distribution
(hist-plot
 (mapcat (fn [[i {:keys [out]}]]
           (let [g (t/grad @out)]
             (println (format "layer %d: mean %+f, std %e" i (t/item (t/mean g)) (t/item (t/std g))))
             (hist-rows g (str "layer " i " (tanh)"))))
         (tanh-layers net)))

;; weights gradient distribution, only the 2d weights
(def weight-params
  (->> (net-params net)
       (map-indexed vector)
       (filter (fn [[_ p]] (= 2 (count (t/shape p)))))))

(hist-plot
 (mapcat (fn [[i p]]
           (let [g (t/grad p)]
             (println (format "weight %s | mean %+f | std %e | grad:data ratio %e" (t/shape p)
                              (t/item (t/mean g)) (t/item (t/std g)) (/ (t/item (t/std g)) (t/item (t/std p)))))
             (hist-rows g (str i " " (t/shape p)))))
         weight-params))

;; update to data ratio over time, a good ratio hovers around 1e-3 (the -3 line)
(let [ud (:ud stats)]
  (-> (tc/dataset (concat
                   (for [[i p] weight-params
                         [step row] (map-indexed vector ud)]
                     {:step step :ratio (nth row i) :series (str i " " (t/shape p))})
                   [{:step 0 :ratio -3 :series "1e-3"}
                    {:step (dec (count ud)) :ratio -3 :series "1e-3"}]))
      (plotly/base {:=x :step :=y :ratio :=color :series :=width 1200})
      (plotly/layer-line)))

;; Loss with the batch norm layers in evaluation mode, which use the running statistics.
(defn net-split-loss
  [net split]
  (set-training! (:layers net) false)
  (let [loss (ctx/no-grad
              (let [[X Y] (splits split)]
                (t/item (t/cross-entropy-f (net-logits net X) Y))))]
    (set-training! (:layers net) true)
    (println split loss)
    loss))

(net-split-loss net :train)
(net-split-loss net :val)

(do (set-training! (:layers net) false)
    (let [names (sample-names #(net-logits net %) 20)]
      (set-training! (:layers net) true)
      names))
