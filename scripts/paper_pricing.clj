;; Supply benchmark pricing for models released after the v0.1.0 runtime.
(require '[clojure.edn :as edn]
         '[clojure.java.io :as io]
         '[spell.provider :as provider])
(let [overlay (-> (io/file (.getParentFile (io/file *file*)) "paper-model-pricing.edn")
                  slurp edn/read-string)]
  (alter-var-root #'provider/default-costs #(merge % overlay)))
