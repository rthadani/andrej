(ns autograd.ops
  (:refer-clojure :exclude [+ * - /])
  (:require [fastmath.random :as frandom]))

(defprotocol Ops 
  (+ [v1 v2])
  (* [v1 v2])
  (tanh [v])
  (** [v x])
  (- [v1 v2])
  (/ [v1 v2])
  (backward [v])
  (gradient [v])
  (set-gradient [_ g]))

(declare value)

(defrecord Value
  [data children op label grad exp]
  Ops
  (+ [this other]
    (let [other (if (not= (type other) Value) (value other) other)]
      (->Value (clojure.core/+ (:data this) (:data other)) [this other] "+" "" (atom 0.0) 0.0)))
  (* [this other]
    (let [other (if (not= (type other) Value) (value other) other)]
      (->Value (clojure.core/* (:data this) (:data other)) [this other] "*" "" (atom 0.0) 0.0)))
  (- [this other]
    (+ this (* -1.0 other)))
  (/ [this other]
    (* this (** other -1)))
  (** [this x]
    (->Value (Math/pow(:data this) x) [this] "**" "" (atom 0.0) x))
  (tanh [this]
    (let [x data
          e**2x (Math/exp (* x 2))
          t (/ (dec e**2x) (inc e**2x))]
      (->Value t [this] "tanh" label (atom 0.0) 0.0)))

  (backward [this]
    (cond 
      (= op "+") 
      (let [[v1 v2] children
            _ (swap! (:grad v1) + (* 1.0 @grad))
            child1 (backward v1)
            _ (swap! (:grad v2) + (* 1.0 @grad))
            child2 (backward v2)]
        (assoc this :children [child1 child2]))
      (= op "*") 
      (let [[v1 v2] children
            _ (swap! (:grad v1) + (* (:data v2) @grad))
            child1 (backward v1)
            _ (swap! (:grad v2) + (* (:data v1) @grad))
            child2 (backward v2) ]
        (assoc this :children [child1 child2]))
      (= op "tanh")
      (do (swap! (:grad (first children)) + (- 1 (Math/pow data 2)))
          (backward (first children)))
      (= op "**")
      (do 
        (swap! (:grad (first children)) + (* (* exp (** data (dec exp))) @grad)) 
        (backward (first children)))
      :else this))
  (gradient [_]
    @grad)
  (set-gradient [this g] 
    (reset! grad g)
    this))

(defn value
  ([data]
   (value data [] "" "" 0))
  ([data label]
   (value data [] "" label 0))
  ([data label grad]
   (value data [] "" label grad))
  ([data children op label grad]
    (if (= ( type data) Value) 
      data 
      (->Value data children op label (atom grad) 0.0))))

(extend-type Number
  Ops
  (+ [this other] (if (= (type other) Value) (+ (value this) other) (clojure.core/+ this other)))
  (* [this other] (if (= (type other) Value) (* (value this) other) (clojure.core/* this other)))
  (** [this x] (Math/pow this x))
  (- [this other] (if (= (type other) Value) (- (value this) other) (clojure.core/- this other)))
  (/ [this other] (if (= (type other) Value) (/ (value this) other) (clojure.core// this other))))

(extend-type Double
  Ops
  (+ [this other] (if (= (type other) Value) (+ (value this) other) (clojure.core/+ this other)))
  (* [this other] (if (= (type other) Value) (* (value this) other) (clojure.core/* this other)))
  (** [this x] (Math/pow this x))
  (- [this other] (if (= (type other) Value) (- (value this) other) (clojure.core/- this other))) 
  (/ [this other] (if (= (type other) Value) (/ (value this) other) (clojure.core// this other))))


(defprotocol NeuronOps
  (activate [_]))

(defrecord Neuron
  [inputs weights bias]
  NeuronOps
  (activate [_]
    (let [weighted-inputs (map * (map value inputs) (map value weights))]
      (tanh (reduce + bias weighted-inputs)))))

(defn neuron
  [inputs]
  (let [weights (map (fn [_] (frandom/frand -1.0 1.0)) ( range (count inputs))) 
        bias (frandom/frand -1.0 1.0)]
    (->Neuron inputs weights bias)))

#_(:data (activate (neuron [2.0 3.0])))
(defprotocol LayerOps
  (forward [_]))

(defrecord Layer
  [neurons num-outputs]
  LayerOps
  (forward [_]
    (let [output (map activate neurons)]
     (if (= 1 num-outputs)
       (first output)
       output))))

(defn layer
  [inputs num-outputs]
  (->Layer (for [_ (range num-outputs)] (neuron inputs)) num-outputs))

#_(map :data (forward (layer [2.0 3.0] 3)))

(defn mlp
  []
  )
