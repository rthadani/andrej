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
      (->Value (clojure.core/+ data (:data other)) [this other] "+" "" (atom 0.0) 0.0)))
  (* [this other]
    (let [other (if (not= (type other) Value) (value other) other)]
      (->Value (clojure.core/* data (:data other)) [this other] "*" "" (atom 0.0) 0.0)))
  (- [this other]
    (+ this (* -1.0 other)))
  (/ [this other]
    (* this (** other -1)))
  (** [this x]
    (->Value (Math/pow data x) [this] "**" "" (atom 0.0) x))
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
          (assoc this :children [(backward (first children))]))
      (= op "**")
      (do 
        (swap! (:grad (first children)) + (* (* exp (** data (dec exp))) @grad)) 
        (assoc this :children [(backward (first children))]))
      :else this))
  (gradient [_]
    @grad)
  (set-gradient [this g] 
    (reset! grad g)
    this))

(defn value
  ([data]
   (value data [] "" "" 0.0))
  ([data label]
   (value data [] "" label 0.0))
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

(defprotocol Parameters
  (parameters [_])
  (update-parameters [_ step]))
(defprotocol NeuronOps
  (activate [_ inputs]))

(defrecord Neuron
  [weights bias]
  NeuronOps
  (activate [_ inputs]
    (let [weighted-inputs (map * (map value inputs) weights)]
      (tanh (reduce + bias weighted-inputs))))
  Parameters
  (parameters [_]
    (conj weights bias))
  (update-parameters [_ step]
    (->Neuron (map  #(update % :data clojure.core/+ (clojure.core/* step @(:grad %))) weights) (update bias :data  + (* step @(:grad bias))))))

(defn neuron
  [num-inputs]
  (let [weights (map (fn [_] (frandom/frand -1.0 1.0)) (range num-inputs)) 
        bias (frandom/frand -1.0 1.0)]
    (->Neuron (map value weights) (value bias))))

(defprotocol LayerOps
  (forward [_ inputs]))

(defrecord Layer
  [neurons num-outputs]
  LayerOps
  (forward [_ inputs]
    (let [output (map #(activate % inputs) neurons)]
     (if (= 1 num-outputs)
       (first output)
       output)))
  Parameters
  (parameters [_]
    (mapcat parameters neurons))
  (update-parameters [_ step]
    (->Layer (map #(update-parameters % step) neurons) num-outputs)))

(defn layer
  [num-inputs num-outputs]
  (->Layer (for [_ (range num-outputs)] (neuron num-inputs)) num-outputs))

(defrecord MLP [layers]
  LayerOps
  (forward [_ inputs]
    (reduce (fn [nn layer] 
              (forward layer nn)) 
          inputs 
          layers))
  Parameters
  (parameters [_]
     (mapcat parameters layers))
  (update-parameters [_ step]
     (->MLP (map #(update-parameters % step) layers))))

(defn mlp
  [input-size layer-sizes]
  (println(cons input-size layer-sizes) (partition 2 1 (cons input-size layer-sizes)))
  (->MLP (map (fn [[i o]] (layer i o)) 
              (partition 2 1 (cons input-size layer-sizes)))))

