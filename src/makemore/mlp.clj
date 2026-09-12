(ns makemore.mlp
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

(count words)
(take 8 words)

(def characters (->> words (str/join "") set sort))
(def stoi  (into (sorted-map) (-> (map-indexed (fn [i c] [c (inc i)]) characters) 
                        (conj [\. 0]))))
(def itos (->> stoi (map (fn [[k v]] [v k])) (into (sorted-map))))

(defn make-xs-ys
  [block-size word]
  (->> (map-indexed (fn[i c] 
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

(->> words 
     (take 5)
     (into [] (block-transform 3)))

(defn make-tensors
  [block-size words]
  (let [tr (fn
             ([] [[] []] )
             ([result] result) 
             ([[xs ys] [chars ch2]]  [(conj xs (map #(stoi %) chars)) (conj ys (stoi ch2))]))
        [xs ys] (transduce (block-transform block-size) tr [[] []] words)]
    [(torch/tensor xs) (torch/tensor ys)]))

(let [[X Y] (make-tensors 3 (take 5 words))
      C (torch/randn [27 2])
      emb (py/get-item C [X])
      W1 (torch/randn [6 100])
      b1 (torch/randn [100])
      h (-> (torch/mm (py. emb view -1 6) W1) (py. __add__ b1) (torch/tanh))
      W2 (torch/randn [100 27])
      b2 (torch/randn 27)
      logits (-> (torch/mm h W2) (py. __add__ b2))
      #_#_counts (py. logits exp)
      #_#_prob (py. counts __div__ (py. counts sum 1 :keepdims true))
      #_#_loss (py.. (py/get-item prob [(torch/arange 32) Y]) log mean __neg__)
      loss (py. F cross_entropy logits Y)
      parameters [C W1 b1 W2 b2]] ;More efficient to do this instead since no intermediate tensors created and uses clustered operations and backward pass is efficient since it uses "fused kernel" operations to calculate the derivative using previously saved values. Numerically well behaved. For very large value exp for logits will be inf 
  (println (py.- X shape) (py.- X dtype) (py.- Y shape) (py.- Y dtype))
  #_(println X) 
  #_(println Y)
  (println (py.- emb shape))
  (println (py/get-item X [13 2]))
  (println (py/get-item (py/get-item C [X]) [13 2]))
  (println (py/get-item C [1]))
  (println (-> (torch/unbind emb 1) (torch/cat 1) (py.- shape)))
  (println (py.- h shape))
  (println (py.- logits shape))
  #_(println logits)
  #_(println (py.- prob shape))
  #_(println (py. (py/get-item prob [0]) sum))
  #_(println loss torch-loss)
  (println loss)
  (println (apply + (map #(py. % nelement) parameters))))

