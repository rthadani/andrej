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

(declare value build-topo backward-step)

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
    (let [topo (build-topo this)]
      (doseq [node (reverse topo)]
        (backward-step node))
      this))

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
    (if (= (type data) Value)
      data
      (->Value data children op label (atom grad) 0.0))))

(defn- build-topo
  "Returns nodes in topological order (leaves first, root last)."
  [root]
  (letfn [(visit [node [visited topo]]
            (if (contains? visited node)
              [visited topo]
              (let [visited' (conj visited node)
                    [visited'' topo'] (reduce (fn [acc child] (visit child acc))
                                              [visited' topo]
                                              (:children node))]
                [visited'' (conj topo' node)])))]
    (second (visit root [#{} []]))))

(defn- backward-step
  "Propagates gradient one step from node to its children."
  [node]
  (let [{:keys [op children data grad exp]} node]
    (case op
      "+" (let [[v1 v2] children]
            (swap! (:grad v1) clojure.core/+ @grad)
            (swap! (:grad v2) clojure.core/+ @grad))
      "*" (let [[v1 v2] children]
            (swap! (:grad v1) clojure.core/+ (clojure.core/* (:data v2) @grad))
            (swap! (:grad v2) clojure.core/+ (clojure.core/* (:data v1) @grad)))
      "tanh" (let [[v] children]
               (swap! (:grad v) clojure.core/+ (clojure.core/* (clojure.core/- 1.0 (clojure.core/* data data)) @grad)))
      "**" (let [[v] children]
             (swap! (:grad v) clojure.core/+ (clojure.core/* (clojure.core/* exp (Math/pow (:data v) (dec exp))) @grad)))
      nil)))

(extend-type Number
  Ops
  (+ [this other] (if (= (type other) Value) (+ (value this) other) (clojure.core/+ this other)))
  (* [this other] (if (= (type other) Value) (* (value this) other) (clojure.core/* this other)))
  (** [this x] (Math/pow this x))
  (- [this other] (if (= (type other) Value) (- (value this) other) (clojure.core/- this other)))
  (/ [this other] (if (= (type other) Value) (/ (value this) other) (clojure.core// this other)))
  (tanh [this] (tanh (value this)))
  (backward [this] this)
  (gradient [_] 0.0)
  (set-gradient [this _] this))

(extend-type Double
  Ops
  (+ [this other] (if (= (type other) Value) (+ (value this) other) (clojure.core/+ this other)))
  (* [this other] (if (= (type other) Value) (* (value this) other) (clojure.core/* this other)))
  (** [this x] (Math/pow this x))
  (- [this other] (if (= (type other) Value) (- (value this) other) (clojure.core/- this other)))
  (/ [this other] (if (= (type other) Value) (/ (value this) other) (clojure.core// this other)))
  (tanh [this] (tanh (value this)))
  (backward [this] this)
  (gradient [_] 0.0)
  (set-gradient [this _] this))

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
    (conj (vec weights) bias))
  (update-parameters [_ step]
    (->Neuron (mapv #(update % :data clojure.core/+ (clojure.core/* step @(:grad %))) weights)
              (update bias :data clojure.core/+ (clojure.core/* step @(:grad bias))))))

(defn neuron
  [num-inputs]
  (let [weights (mapv (fn [_] (value (frandom/frand -1.0 1.0))) (range num-inputs))
        bias    (value (frandom/frand -1.0 1.0))]
    (->Neuron weights bias)))

(defprotocol LayerOps
  (forward [_ inputs]))

(defrecord Layer
  [neurons num-outputs]
  LayerOps
  (forward [_ inputs]
    (let [output (mapv #(activate % inputs) neurons)]
      (if (= 1 num-outputs)
        (first output)
        output)))
  Parameters
  (parameters [_]
    (mapcat parameters neurons))
  (update-parameters [_ step]
    (->Layer (mapv #(update-parameters % step) neurons) num-outputs)))

(defn layer
  [num-inputs num-outputs]
  (->Layer (mapv (fn [_] (neuron num-inputs)) (range num-outputs)) num-outputs))

(defrecord MLP [layers]
  LayerOps
  (forward [_ inputs]
    (reduce (fn [acts layer]
              (forward layer acts))
            inputs
            layers))
  Parameters
  (parameters [_]
    (mapcat parameters layers))
  (update-parameters [_ step]
    (->MLP (mapv #(update-parameters % step) layers))))

(defn mlp
  [input-size layer-sizes]
  (println (cons input-size layer-sizes) (partition 2 1 (cons input-size layer-sizes)))
  (->MLP (mapv (fn [[i o]] (layer i o))
               (partition 2 1 (cons input-size layer-sizes)))))
