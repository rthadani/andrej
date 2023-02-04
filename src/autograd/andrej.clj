(ns autograd.andrej
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as td]
            [nextjournal.clerk :as clerk]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]
            [aerial.hanami.core :as hmi]
            [dorothy.core :as dot]
            [autograd.ops :refer [value + * tanh backward gradient set-gradient ] :as ops]
            [clojure.java.io :as io]
            [fastmath.core :as math])
   (:import [javax.imageio ImageIO]))

#_(clerk/serve! 
  {:watch-paths ["src/autograd"]
   :browse? true})

(defn line-chart
  [xy]
  (hc/xform ht/line-chart
            :DATA xy 
            :X :x
            :Y :y))

(defn f
  [x]
  (-> x
      (Math/pow 2)
      (* 3)
      (- (* x 4))
      (+ 5)))

(f 3)

(->> (range -5 5 0.25)
     (reduce (fn [a x] (conj a {:x x :y (f x)}) ) [])
     (line-chart)
     (clerk/vl))

;differentiate
(let [h  0.0000001
      x  3.0]
  (/ (- (f (+ x h)) (f x)) h))

(let [h  0.0000001
      x  -3.0]
  (/ (- (f (+ x h)) (f x)) h))

(let [h  0.0000001
      x  (/ 2 3)]
  (/ (- (f (+ x h)) (f x)) h))

(let [a 2.0
      b -3.0
      c 10
      h 0.0001
      d1 (+ (* a b) c)
      a (+ a h)
      d2 (+ (* a b) c) ]
  (println d1 d2)
  (/ (- d2 d1) h))

(defn trace-root
  ([root]
   (trace-root root #{} #{}))
  ([root nodes edges]
    (if (empty? (:children root))
      [(conj nodes root) edges]
    (reduce (fn [[nodes edges] c] (trace-root c nodes (conj edges [c root]))) [(conj nodes root) edges] (:children root)))))

(defn draw-dot
  [root file-name]
  (let [[nodes edges] (trace-root root)
        [dot-nodes dot-edges] (reduce (fn [result {:keys [label data op] :as n}]
                                        (let [uid (str (hash n))
                                              value-node [uid {:label  (format "%s | data %.4f | grad %.4f" label (float data) (float (gradient n))) :shape :record}]
                                              op-node [(str uid op) {:label op}]]
                                          (if (not-empty op)
                                            [(conj (first result) value-node op-node) (conj (second result) [(first op-node) uid])]
                                            [(conj (first result) value-node) (second result)]))) 
                                      [[] []] nodes) 
        dot-edges (reduce (fn [e [n1 n2]]  
                            (conj e [(str (hash n1)) (str (hash n2) (:op n2))])) 
                          dot-edges edges)]
    (-> (dot/digraph  (cons (dot/graph-attrs {:rankdir :LR}) (concat dot-edges dot-nodes)))
        dot/dot
        (dot/save! file-name {:format :png}))))

;^::clerk/no-cache 
(def L
  (let [a (value 2.0 "a" (* -3 -2.0)) ; da/dl = da/de(e = a*b therefore b) * de/dl (-2)
      b (value -3.0 "b" (* 2 -2.0))
      c (value 10.0 "c" -2.0) ;by the chain rule dl/dc = dl/dd(-2) * dd/dc (1)
      e (-> (assoc (* a b) :label "e") (set-gradient -2.0)) ;by the chain rule de/dl = de/dd(-2) * dd/dl(1)
      _ (println "HERE")
      d (-> (assoc (+ e c) :label "d") (set-gradient -2.0))
      f (value -2.0 "f" 4.0)]
  (-> (assoc (* d f) :label "L") (set-gradient 1.0))))

^::clerk/no-cache (draw-dot L "out.png")
^::clerk/no-cache (ImageIO/read  (io/file "out.png"))

(defn lol
  []
  (let [a (value 2 "a")
      b (value -3 "b")
      c (value 10 "c")
      e (assoc (* a b) :label "e")
      d (assoc (+ e c) :label "d")
      f (value -2 "f" )
      L1 (:data (* d f))
      
      h 0.001

      a (value 2 "a")
      b (value (+ -3 h) "b")
      c (value 10 "c")
      e (assoc (* a b) :label "e")
      d (assoc (+ e c) :label "d")
      f (value -2 "f" )
      L2 (:data (* d f))]
  (/ (- L2 L1) h)))

(lol)

;step
;nudge all the inputs by a small amount to see effect on output
(let [a (value 2 "a" (* -3 -2.0)) 
      a (update a :data #(+ % (* 0.01 (gradient a))))
      b (value -3 "b" (* 2 -2.0))
      b (update b :data #(+ % (* 0.01 (gradient b))))
      c (value 10 "c" -2.0) 
      c (update c :data #(+ % (* 0.01 (gradient c))))
      e (-> (assoc (* a b) :label "e") (set-gradient -2.0)) 
      d (-> (assoc (+ e c) :label "d") (set-gradient -2.0))
      f (value -2 "f" 4.0)
      f (update f :data #(+ % (* 0.01 (gradient f)))) ]
  (:data (* d f))) ; -7.286496

;tan-h activation
(->> (range -5 5 0.2)
     (reduce (fn [a x] (conj a {:x x :y (math/tanh x)}) ) [])
     (line-chart)
     (clerk/vl))

;;nn sample handroll back prop
^::clerk/no-cache
(def nn (let [x1 (value 2.0 "x1" -1.5) 
      x2 (value 0.0 "x2" 0.5) ;x2 grad = w2.data * grad x2w2 
      w1 (value -3.0 "w1" 1.0)
      w2 (value 1.0 "w2" 0.0) ;w2 grad = x2.data * grad x2w2
      b (value 6.8813735870195432 "b" 0.5) ; + distributes the gradient (backprop grad n to all parents)
      x1w1 (-> (assoc (* x1 w1) :label "x1w1") (set-gradient 0.5)) ; + distributes the gradient (backprop grad x1w1 + x2w2 to all parents)
      x2w2 (-> (assoc (* x2 w2) :label "x2w2") (set-gradient 0.5)) ; + distributes the gradient (backprop grad x1w1 + x2w2 to all parents)
      x1w1+x2w2 (-> (assoc (+ x1w1 x2w2) :label "x1*w1 + x2*w2") (set-gradient 0.5)) ; + distributes the gradient (backprop grad n to all parents)
      n (-> (assoc (+ x1w1+x2w2 b) :label "n") (set-gradient 0.5)) ;do/dn (d tanh(o)/dn = (1 - tanh(o) ** 2)
      o (-> (assoc (tanh n) :label "o") (set-gradient 1.0))]
      o))

^::clerk/no-cache (draw-dot nn "out2.png")
^::clerk/no-cache (ImageIO/read  (io/file "out2.png"))

;;nn with back prop implemented
^::clerk/no-cache 
(def nn1 (let [x1 (value 2.0 "x1") 
      x2 (value 0.0 "x2") 
      w1 (value -3.0 "w1")
      w2 (value 1.0 "w2") 
      b (value 6.8813735870195432 "b") 
      x1w1 (assoc (* x1 w1) :label "x1w1") 
      x2w2 (assoc (* x2 w2) :label "x2w2") 
      x1w1+x2w2 (assoc (+ x1w1 x2w2) :label "x1*w1 + x2*w2") 
      n (assoc (+ x1w1+x2w2 b) :label "n") 
      o (-> (assoc (tanh n) :label "o") (set-gradient 1.0))]
      (backward o)))

^::clerk/no-cache (draw-dot nn1 "out3.png")
^::clerk/no-cache (ImageIO/read  (io/file "out3.png"))

^::clerk/no-cache 
(def bug (let [a (value 3.0 "a")
               b (assoc (+ a a) :grad (atom 1.0) :label "b") ]
  (backward b)))

^::clerk/no-cache (draw-dot bug "out4.png")
^::clerk/no-cache (ImageIO/read  (io/file "out4.png"))

(def bug2
  (let [a (value -2.0 "a")
        b (value 3.0 "b")
        d (assoc (* a b) :label "d")
        e (assoc (+ a b) :label "e")
        f (-> (assoc (* d e) :label "f") (set-gradient 1.0))]
    (backward f)))

^::clerk/no-cache (draw-dot bug2 "out5.png")
^::clerk/no-cache (ImageIO/read  (io/file "out5.png"))


(:data (ops/activate (ops/neuron [2.0 3.0])))

(map :data (ops/forward (ops/layer [2.0 3.0] 3)))

^::clerk/no-cache
(def mlp-nn (ops/mlp [2.0 3.0 -1.0] [4 4 1]))
^::clerk/no-cache (draw-dot mlp-nn "out6.png")
^::clerk/no-cache (ImageIO/read  (io/file "out6.png"))
