(ns makemore.mlp
  (:require [clojure.string :as str]
            [clj-pytorch.functional :as t]
            [clj-pytorch.tensor :as tensor]
            [nextjournal.clerk :as clerk]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]))

#_(clerk/serve! 
  {:watch-paths ["src/makemore"]
   :browse? true})

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
    [(t/tensor xs) (t/tensor ys)]))

(let [[X Y] (make-tensors 3 (take 5 words))
      C (t/randn [27 2])
      emb (-> (t/index-select C 0 (t/view X [-1])) (t/view (conj (t/shape X) 2)))
      W1 (t/randn [6 100])
      b1 (t/randn [100])
      h (-> (t/matmul (t/view emb [-1 6]) W1) (t/add b1) (t/tanh-f))
      W2 (t/randn [100 27])
      b2 (t/randn [27])
      logits (t/add (t/matmul h W2) b2)
      #_#_counts (t/exp logits)
      #_#_prob (t/div counts (t/sum counts 1 true))
      #_#_loss (t/neg (t/mean (t/log (t/gather prob 1 (t/unsqueeze Y 1)))))
      loss (t/cross-entropy-f logits Y)
      parameters [C W1 b1 W2 b2]] ;More efficient to do this instead since no intermediate tensors created and uses clustered operations and backward pass is efficient since it uses "fused kernel" operations to calculate the derivative using previously saved values. Numerically well behaved. For very large value exp for logits will be inf 
  (println (t/shape X) (t/dtype X) (t/shape Y) (t/dtype Y))
  #_(println X) 
  #_(println Y)
  (println (t/shape emb))
  (println (t/tensor-get X 13 2))
  (println (t/select (t/select emb 0 13) 0 2))
  (println (t/select C 0 1))
  (println (-> (t/cat (map #(t/select emb 1 %) (range (nth (t/shape emb) 1))) :dim 1) t/shape))
  (println (t/shape h))
  (println (t/shape logits))
  #_(println logits)
  #_(println (t/shape prob))
  #_(println (t/sum (t/select prob 0 0)))
  #_(println loss torch-loss)
  (println loss)
  (println (apply + (map t/numel parameters))))
