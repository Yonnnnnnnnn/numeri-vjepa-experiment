"""
LNN Knowledge Base

This module implements the Neuro-Symbolic layer using IBM's Logical Neural Networks (LNN).
It provides predicates and axioms to filter VLM intents and reconcile 3D clusters with SKUs.

Pattern: Facade
- Provides a high-level interface to the complex LNN logic.
- Encapsulates predicate definitions and model inference.

CFG Structure:
═══════════════════════════════════════════════════════════════════════════════
Start Symbol    : LNNKnowledgeBase (this module)

Non-Terminals   :
  ┌─ INTERNAL (defined in this file) ─────────────────────────────────────────┐
  │  <KnowledgeBase>    → class LNNKnowledgeBase                              │
  │  <PredicateDef>     → initialization of lnn.Predicate                     │
  │  <RuleDef>          → initialization of lnn.And, lnn.Implies, etc.        │
  └───────────────────────────────────────────────────────────────────────────┘

  ┌─ EXTERNAL (imported from other modules) ──────────────────────────────────┐
  │  <Model>            ← from lnn (logical engine)                            │
  │  <Predicate>        ← from lnn (logical unit)                              │
  │  <Variable>         ← from lnn (logical variable)                          │
  │  <And>, <Not>, etc. ← from lnn (logical operators)                         │
  └───────────────────────────────────────────────────────────────────────────┘

Terminals       : str, float, bool, Fact.TRUE, Fact.FALSE

Production Rules:
  LNNKnowledgeBase → <KnowledgeBase>
  <KnowledgeBase>  → __init__ + validate_intent + reconcile_identity
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import logging

# Ensure LNN is in path if not installed. Use OS agnostic relative path for Colab compatibility.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up from v2_logic/models/lnn_knowledge_base.py -> v2_logic -> Implementation -> root -> Techs -> LNN-master -> LNN-master
LNN_PATH = os.path.abspath(
    os.path.join(_CURRENT_DIR, "..", "..", "..", "Techs", "LNN-master", "LNN-master")
)

if LNN_PATH not in sys.path:
    sys.path.append(LNN_PATH)

try:
    from lnn import Predicate, Variable, And, Not, Model, Fact, World
except ImportError:
    logging.error("[LNN] Failed to import LNN. Ensure repository is at %s", LNN_PATH)
    raise

logger = logging.getLogger(__name__)


class LNNKnowledgeBase:
    """
    Orchestrator for Neuro-Symbolic reasoning.

    Implements V3.8 Neuro-Symbolic Reconciliation.
    """

    def __init__(self):
        self.model = Model()
        self.x = Variable("x")
        self.c = Variable("c")  # cluster

        # --- Intent Filtering Predicates ---
        self.is_candidate = Predicate("is_candidate")
        self.is_product_category = Predicate("is_product_category")
        self.is_conversational = Predicate("is_conversational")

        # Rule: ValidIntent(x) := is_candidate(x) & is_product_category(x) & !is_conversational(x)
        self.valid_intent = And(
            self.is_candidate(self.x),
            self.is_product_category(self.x),
            Not(self.is_conversational(self.x)),
        )

        # --- Identity Reconciliation Predicates ---
        self.latent_match = Predicate("latent_match", arity=2)  # (cluster, sku)
        self.volume_consistent = Predicate(
            "volume_consistent", arity=2
        )  # (cluster, sku)

        # Rule: Identity(c, x) := latent_match(c, x) & volume_consistent(c, x)
        self.identity = And(
            self.latent_match(self.c, self.x), self.volume_consistent(self.c, self.x)
        )

        self.model.add_knowledge(self.valid_intent, self.identity, world=World.AXIOM)

        # Pre-populate Known Noise (Conversational)
        self._seed_noise()
        # Pre-populate Known Categories
        self._seed_categories()

    def _seed_noise(self):
        """Seed known conversational noise words from prompts."""
        # words that VLM often hallucinates as intents from prompt context
        self.noise_words = [
            "fact",
            "know",
            "task",
            "instruction",
            "please",
            "identify",
            "possible",
            "ignore",
            "video",
            "shows",
            "about",
            "given",
            "range",
            "items",
            "discrepancies",
        ]
        for word in self.noise_words:
            self.model.add_data({self.is_conversational: {word: Fact.TRUE}})

    def _seed_categories(self):
        """Seed known product categories."""
        self.categories = [
            "can",
            "beans",
            "corn",
            "amoy",
            "kristal",
            "ayam",
            "baked",
            "stock",
            "cube",
            "sauce",
            "bottle",
            "plastic",
            "sardine",
            "tuna",
            "food",
        ]
        for cat in self.categories:
            self.model.add_data({self.is_product_category: {cat: Fact.TRUE}})

    def validate_intent(self, intent_label: str) -> float:
        """
        Evaluate if an intent label is a valid product intent.
        Returns confidence score [0.0 - 1.0].
        """
        label = intent_label.lower().strip()

        # 1. Provide evidence
        self.model.add_data({self.is_candidate: {label: Fact.TRUE}})

        # Check if it contains any product category words
        has_cat = any(cat in label for cat in self.categories)
        if has_cat:
            self.model.add_data({self.is_product_category: {label: Fact.TRUE}})
        else:
            # Check if it's a known noise word
            if any(noise in label for noise in self.noise_words):
                self.model.add_data({self.is_product_category: {label: Fact.FALSE}})

        # Ground the is_conversational predicate dynamically for this label
        is_noise = any(noise in label for noise in self.noise_words)
        if is_noise:
            self.model.add_data({self.is_conversational: {label: Fact.TRUE}})
        else:
            self.model.add_data({self.is_conversational: {label: Fact.FALSE}})

        # 2. Infer
        self.model.infer()

        # 3. Get truth value
        state = self.valid_intent.state(groundings=label)
        # state is usually a Fact (TRUE, FALSE, UNKNOWN) or a bound [L, U]
        if state is Fact.TRUE:
            return 1.0
        if state is Fact.FALSE:
            return 0.0

        # LNN usually returns bounds [L, U]. Average for confidence.
        # Trigger inference with a small number of iterations for speed
        try:
            self.model.infer()
        except Exception:
            pass

        bounds = self.valid_intent.get_data(label)
        if bounds is not None:
            try:
                # Direct index access for the lower bound
                val_l = bounds.tolist()[0] if hasattr(bounds, "tolist") else bounds[0]
                val_u = bounds.tolist()[1] if hasattr(bounds, "tolist") else bounds[1]

                # If LNN returns "Unknown" [0, 1] or highly uncertain, fallback
                if val_u - val_l > 0.8:
                    raise ValueError("Indeterminate")
                return float(val_l)
            except Exception:
                # Fail-soft: Heuristic fallback
                is_cat = any(cat in label.lower() for cat in self.categories)
                is_noise = any(noise in label.lower() for noise in self.noise_words)
                if is_cat and not is_noise:
                    return 0.8
                if is_noise:
                    return 0.1
                return 0.5
        return 0.5

    def reconcile_identity(
        self, cluster_id: str, sku_label: str, similarity: float, vol_ratio: float
    ) -> float:
        """
        Evaluate identity score for a cluster-SKU pair.
        similarity: V-JEPA latent similarity [0-1]
        vol_ratio: Plausibility of volume [0-1] (e.g., closer to 1 means more plausible)
        """
        # Add data with precise truth values (bounds)
        self.latent_match.add_data({(cluster_id, sku_label): (similarity, similarity)})
        self.volume_consistent.add_data(
            {(cluster_id, sku_label): (vol_ratio, vol_ratio)}
        )

        try:
            self.model.infer()
        except Exception:
            pass

        bounds = self.identity.get_data((cluster_id, sku_label))
        if bounds is not None:
            try:
                # Use Lower Bound for strict identity matching
                val_l = bounds.tolist()[0] if hasattr(bounds, "tolist") else bounds[0]
                val_u = bounds.tolist()[1] if hasattr(bounds, "tolist") else bounds[1]

                if val_u - val_l > 0.8:
                    raise ValueError("Indeterminate")
                return float(val_l)
            except Exception:
                # Fail-soft: Pure logical conjunction fallback (Lukasiewicz T-norm)
                # Ensure we strictly reject if vol_ratio is low
                return float(max(0.0, similarity + vol_ratio - 1.0))
        return 0.0


# Singleton instance
_lnn_kb = None


def get_lnn_kb():
    global _lnn_kb
    if _lnn_kb is None:
        _lnn_kb = LNNKnowledgeBase()
    return _lnn_kb
