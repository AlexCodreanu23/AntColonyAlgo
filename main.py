"""
Ant Colony Algorithm for Unsupervised Word Sense Disambiguation
Based on: Schwab et al., COLING 2012
"Ant Colony Algorithm for the Unsupervised Word Sense Disambiguation of Texts:
 Comparison and Evaluation"

Corpus: SemEval 2007 Task 7 (Coarse-Grained English All-Words)
"""

import os
import math
import random
import re
import copy
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from nltk.corpus import wordnet as wn


print(wn.lemma_from_key("bank%1:14:00::").synset().name())
print(wn.lemma_from_key("bank%1:17:00::").synset().name())
print(wn.lemma_from_key("mouse%1:06:00::").synset().name())
# ─────────────────────────────────────────────
#  SECTION 1 – DATA LOADING (SemEval 2007 Task 7)
# ─────────────────────────────────────────────

class SemEvalInstance:
    """Holds a single annotated word instance from the corpus."""
    def __init__(self, inst_id: str, lemma: str, pos: str,
                 context_words: List[str], gold_senses: List[str]):
        self.inst_id = inst_id          # e.g. "d001.s001.t001"
        self.lemma   = lemma            # e.g. "church"
        self.pos     = pos              # n / v / a / r
        self.context_words = context_words
        self.gold_senses   = gold_senses   # synset keys, may be >1 (coarse)


class SemEvalDocument:
    """One document from the corpus: list of sentences, each a list of tokens."""
    def __init__(self, doc_id: str):
        self.doc_id    = doc_id
        # ordered list of (token_text, inst_id_or_None) per sentence
        self.sentences: List[List[Tuple[str, Optional[str]]]] = []
        # inst_id -> SemEvalInstance
        self.instances: Dict[str, SemEvalInstance] = {}


def load_babelnet_to_wordnet_mapping(mapping_path: str) -> Dict[str, str]:
    """
    Încarcă un fișier TSV: bn_id\twn_synset_name
    ex: bn:00020825n\tperson.n.01
    """
    mapping = {}
    if not os.path.exists(mapping_path):
        return mapping
    with open(mapping_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping

def load_semeval2007_task7(data_xml_path: str, key_path: str) -> List[SemEvalDocument]:
    """
    Parse the SemEval 2007 Task 7 coarse-grained corpus.

    data_xml_path : path to the corpus XML  (e.g. eng-coarse-all-words.xml)
    key_path      : path to the gold-key file (e.g. corpus.gold.key.txt)

    XML structure (simplified):
      <corpus>
        <text id="d001">
          <sentence id="d001.s001">
            <wf  lemma="the"    pos="DET">The</wf>
            <instance id="d001.s001.t001" lemma="church" pos="NN">church</instance>
          </sentence>
        </text>
      </corpus>

    Gold key:  instance_id  sense_key1 [sense_key2 ...]
    """
    # --- 1a. Read gold keys -------------------------------------------------
    gold: Dict[str, List[str]] = {}
    if os.path.exists(key_path):
        with open(key_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    gold[parts[0]] = parts[1:]

    # --- 1b. Parse XML -------------------------------------------------------
    documents: List[SemEvalDocument] = []

    if not os.path.exists(data_xml_path):
        print(f"[WARNING] Corpus file not found: {data_xml_path}")
        print("[INFO]    Falling back to synthetic demo corpus …")
        return _build_synthetic_corpus()

    tree = ET.parse(data_xml_path)
    root = tree.getroot()

    for text_elem in root.iter("text"):
        doc = SemEvalDocument(text_elem.attrib.get("id", "unknown"))

        for sent_elem in text_elem.iter("sentence"):
            sentence_tokens: List[Tuple[str, Optional[str]]] = []

            for child in sent_elem:
                surface = (child.text or "").strip()
                if child.tag == "instance":
                    inst_id = child.attrib["id"]
                    lemma   = child.attrib.get("lemma", surface)
                    pos_raw = child.attrib.get("pos", "NN")
                    pos     = _penn_to_wn_pos(pos_raw)
                    ctx     = _sentence_context(sent_elem)
                    g_senses = gold.get(inst_id, [])
                    inst = SemEvalInstance(inst_id, lemma, pos, ctx, g_senses)
                    doc.instances[inst_id] = inst
                    sentence_tokens.append((surface, inst_id))
                else:
                    sentence_tokens.append((surface, None))

            doc.sentences.append(sentence_tokens)

        documents.append(doc)

    print(f"[Corpus] Loaded {len(documents)} documents, "
          f"{sum(len(d.instances) for d in documents)} instances.")
    return documents


def _penn_to_wn_pos(penn: str) -> str:
    """Convert Penn Treebank POS tag to WordNet POS character."""
    penn = penn.upper()
    if penn.startswith("NN"):  return wn.NOUN
    if penn.startswith("VB"):  return wn.VERB
    if penn.startswith("JJ"):  return wn.ADJ
    if penn.startswith("RB"):  return wn.ADV
    return wn.NOUN


def _sentence_context(sent_elem) -> List[str]:
    """Return all surface words in a sentence element."""
    words = []
    for child in sent_elem:
        w = (child.text or "").strip()
        if w:
            words.append(w.lower())
    return words


def generate_bn_wn_mapping_from_gold(documents, output_path="bn_wn_mapping.tsv"):
    """
    Încearcă să găsească synset WordNet potrivit pentru fiecare BabelNet ID din gold,
    folosind numele lemei și POS ca hint.
    """
    import re
    pos_map = {"n": wn.NOUN, "v": wn.VERB, "r": wn.ADV, "a": wn.ADJ}

    with open(output_path, "w", encoding="utf-8") as out:
        for doc in documents:
            for inst_id, inst in doc.instances.items():
                for key in inst.gold_senses:
                    key = key.strip()
                    if not key.startswith("bn:"):
                        continue
                    m = re.match(r"bn:(\d+)([nvra])$", key)
                    if not m:
                        continue
                    pos_char = m.group(2)
                    pos = pos_map.get(pos_char)
                    if not pos:
                        continue
                    # Caută synset-uri pentru lema cuvântului cu POS-ul corect
                    synsets = wn.synsets(inst.lemma, pos=pos)
                    if synsets:
                        # Primul synset ca aproximare (nu e perfect)
                        out.write(f"{key}\t{synsets[0].name()}\n")
# ─────────────────────────────────────────────
#  SYNTHETIC CORPUS (fallback when no file)
# ─────────────────────────────────────────────



def _build_synthetic_corpus() -> List[SemEvalDocument]:
    """Build a small synthetic corpus for demonstration / testing."""
    sentences_raw = [
        [("The",    None),  ("bank",   "d000.s000.t000"), ("can",  None),
         ("guarantee", None), ("deposits", None)],
        [("The",    None),  ("river",  None), ("bank",  "d000.s001.t000"),
         ("was",    None),  ("steep",  None)],
        [("The",    None),  ("mouse",  "d000.s002.t000"), ("clicked", None)],
        [("A",      None),  ("church", "d000.s003.t000"), ("stands", None),
         ("nearby", None)],
    ]
    gold_map = {
        "d000.s000.t000": ["bank%1:14:01::"],   # financial institution
        "d000.s001.t000": ["bank%1:17:00::"],   # river bank
        "d000.s002.t000": ["mouse%1:05:00::"],  # computer mouse
        "d000.s003.t000": ["church%1:14:00::"], # building
    }
    lemma_map = {
        "d000.s000.t000": ("bank", wn.NOUN),
        "d000.s001.t000": ("bank", wn.NOUN),
        "d000.s002.t000": ("mouse", wn.NOUN),
        "d000.s003.t000": ("church", wn.NOUN),
    }
    doc = SemEvalDocument("d000")
    all_words = [w for sent in sentences_raw for w, _ in sent]
    for sent in sentences_raw:
        doc.sentences.append(sent)
        for _, iid in sent:
            if iid:
                lemma, pos = lemma_map[iid]
                inst = SemEvalInstance(iid, lemma, pos,
                                       [w.lower() for w, _ in sent],
                                       gold_map.get(iid, []))
                doc.instances[iid] = inst
    return [doc]


# ─────────────────────────────────────────────
#  SECTION 2 – LESK MEASURE
# ─────────────────────────────────────────────

class LeskMeasure:
    """
    Extended Lesk (ExtLesk) as described in the paper.
    Bag-of-words overlap between synset glosses, including glosses of related
    synsets (hypernyms, hyponyms, meronyms, holonyms, similar, also).
    Uses integer indexing to achieve O(m) comparison instead of O(mn).
    """

    def __init__(self):
        # word string -> unique integer id
        self._word_index: Dict[str, int] = {}
        self._next_id = 0
        # cache: synset.name() -> frozenset of word-ids
        self._synset_vector_cache: Dict[str, frozenset] = {}

    # ── private helpers ──────────────────────────────────────────────────────

    def _word_id(self, word: str) -> int:
        w = word.lower().strip()
        if w not in self._word_index:
            self._word_index[w] = self._next_id
            self._next_id += 1
        return self._word_index[w]

    def _tokenise(self, text: str) -> List[int]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return [self._word_id(t) for t in tokens]

    def _get_related_synsets(self, synset) -> List:
        """Collect the synset itself plus all directly linked synsets."""
        related = [synset]
        for fn in [synset.hypernyms, synset.hyponyms,
                   synset.member_meronyms, synset.part_meronyms,
                   synset.substance_meronyms, synset.member_holonyms,
                   synset.similar_tos, synset.also_sees]:
            try:
                related.extend(fn())
            except Exception:
                pass
        return related

    def synset_vector(self, synset) -> frozenset:
        """Return the indexed word-id frozenset for a synset (cached)."""
        key = synset.name()
        if key not in self._synset_vector_cache:
            ids: Set[int] = set()
            for ss in self._get_related_synsets(synset):
                ids.update(self._tokenise(ss.definition() or ""))
                for ex in ss.examples():
                    ids.update(self._tokenise(ex))
            self._synset_vector_cache[key] = frozenset(ids)
        return self._synset_vector_cache[key]

    def score(self, synset_a, synset_b) -> int:
        """
        ExtLesk score: size of intersection of the two indexed word-id sets.
        Complexity: O(m) where m = len(larger set).
        """
        va = self.synset_vector(synset_a)
        vb = self.synset_vector(synset_b)
        return len(va & vb)

    def score_vector_synset(self, vec: frozenset, synset) -> int:
        """Score between an arbitrary word-id frozenset and a synset vector."""
        vb = self.synset_vector(synset)
        return len(vec & vb)

    def score_two_vectors(self, va: frozenset, vb: frozenset) -> int:
        return len(va & vb)


# ─────────────────────────────────────────────
#  SECTION 3 – GRAPH / TREE NODES
# ─────────────────────────────────────────────

class Node:
    """
    A node in the ACA graph.
    Can be a 'plain' node (text / sentence / word structural node)
    or a 'nest' node (word sense).
    """
    _id_counter = 0

    def __init__(self, node_type: str, label: str,
                 energy: float = 0.0, odour_length: int = 100):
        Node._id_counter += 1
        self.node_id   : int   = Node._id_counter
        self.node_type : str   = node_type   # 'text'|'sentence'|'word'|'nest'
        self.label     : str   = label

        # Energy stored on this node
        self.energy    : float = energy

        # Odour vector: frozenset of word-ids deposited by passing ants
        # For nest nodes this is fixed (synset definition); for plain nodes
        # it is built up dynamically.
        self.odour_vector : Set[int] = set()
        self.odour_length : int      = odour_length   # max components LV

        # For nest nodes only
        self.synset        = None          # wn.Synset object
        self.word_lemma    : str   = ""
        self.word_pos      : str   = ""
        self.inst_id       : str   = ""    # corpus instance id

        # Edges: neighbour node_id -> pheromone level
        self.edges : Dict[int, float] = {}

        # Ants currently visiting this node (this cycle)
        self.visiting_ants : List['Ant'] = []

        # Vote count across cycles (for majority-vote strategy)
        self.vote_count    : int   = 0

    @property
    def is_nest(self) -> bool:
        return self.node_type == "nest"

    def add_neighbour(self, other: 'Node', pheromone: float = 0.0):
        self.edges[other.node_id] = pheromone
        other.edges[self.node_id] = pheromone

    def deposit_odour(self, ant_odour: frozenset, delta_v_pct: float):
        """
        An ant deposits a fraction (delta_v_pct / 100) of its odour components
        onto this plain node (chosen uniformly at random from ant's odour).
        Plain-node odour size is capped at self.odour_length.
        """
        if self.is_nest:
            return  # nest odour is immutable
        n_deposit = max(1, int(len(ant_odour) * delta_v_pct / 100.0))
        sample = random.sample(list(ant_odour), min(n_deposit, len(ant_odour)))
        for word_id in sample:
            if len(self.odour_vector) < self.odour_length:
                self.odour_vector.add(word_id)
            else:
                # Replace a random existing component
                victim = random.choice(list(self.odour_vector))
                self.odour_vector.discard(victim)
                self.odour_vector.add(word_id)

    def __repr__(self):
        return (f"Node(id={self.node_id}, type={self.node_type}, "
                f"label={self.label!r}, E={self.energy:.2f})")


class Bridge:
    """
    A bridge is a dynamic edge connecting two nest nodes (potential friends).
    It collapses when its pheromone reaches 0.
    """
    def __init__(self, nest_a: Node, nest_b: Node, pheromone: float = 0.01):
        self.nest_a_id : int   = nest_a.node_id
        self.nest_b_id : int   = nest_b.node_id
        self.pheromone : float = pheromone
        self.alive     : bool  = True

    def deposit(self, amount: float):
        self.pheromone += amount

    def evaporate(self, delta: float):
        """Evaporate pheromone; mark bridge dead if it reaches 0."""
        self.pheromone *= (1.0 - delta)
        if self.pheromone <= 1e-6:
            self.pheromone = 0.0
            self.alive     = False

    def __repr__(self):
        return (f"Bridge({self.nest_a_id}<->{self.nest_b_id}, "
                f"phi={self.pheromone:.4f}, alive={self.alive})")


# ─────────────────────────────────────────────
#  SECTION 4 – ANT
# ─────────────────────────────────────────────

class Ant:
    """
    A single ant agent.

    Modes
    -----
    'explore'  – seeking energy, avoids pheromone on edges,
                 attracted to high-energy nodes.
    'return'   – carrying energy home, follows pheromone + odour similarity.
    """
    _id_counter = 0

    def __init__(self, mother_nest: Node, lifespan: int,
                 e_max: float, odour_vector: frozenset):
        Ant._id_counter += 1
        self.ant_id       : int          = Ant._id_counter
        self.mother_nest  : Node         = mother_nest
        self.current_node : Node         = mother_nest
        self.lifespan_rem : int          = lifespan     # ω remaining cycles
        self.energy       : float        = 0.0
        self.e_max        : float        = e_max
        self.mode         : str          = "explore"    # 'explore' | 'return'
        # Odour = indexed word-id set from mother nest definition (immutable)
        self.odour_vector : frozenset    = odour_vector
        # Path taken this cycle (node_ids) for logging
        self.path         : List[int]    = [mother_nest.node_id]
        self.alive        : bool         = True

    def pick_up_energy(self, node: Node, e_a: float):
        """
        Collect e_a units of energy from node (capped at node.energy and
        what the ant can still carry).
        """
        available   = node.energy
        can_carry   = self.e_max - self.energy
        collected   = min(e_a, available, can_carry)
        node.energy = max(0.0, node.energy - collected)
        self.energy += collected

    def return_probability(self) -> float:
        """
        P(return) = (E(f) / E_max)^3
        When energy == E_max this equals 1.0 (guaranteed switch to return).
        """
        return (self.energy / self.e_max) ** 3 if self.e_max > 0 else 0.0

    def maybe_switch_to_return(self):
        if self.mode == "explore":
            if random.random() < self.return_probability():
                self.mode = "return"

    def tick(self):
        """Decrement lifespan by one cycle."""
        self.lifespan_rem -= 1
        if self.lifespan_rem <= 0:
            self.alive = False

    def __repr__(self):
        return (f"Ant(id={self.ant_id}, nest={self.mother_nest.label!r}, "
                f"mode={self.mode}, E={self.energy:.2f}, "
                f"life={self.lifespan_rem})")


# ─────────────────────────────────────────────
#  SECTION 5 – GRAPH BUILDER
# ─────────────────────────────────────────────

class ACAGraph:
    """
    Builds and maintains the ACA environment graph.

    Structure (per paper, Fig. 1):
      text_node
        └─ sentence_node  (one per sentence)
             └─ word_node  (one per ambiguous word)
                  └─ nest_node  (one per candidate sense)

    Bridges between nest nodes of different words are added dynamically.
    """

    def __init__(self, lesk: LeskMeasure,
                 E0: float = 20.0, odour_length: int = 100):
        self.lesk          = lesk
        self.E0            = E0
        self.odour_length  = odour_length

        self.nodes      : Dict[int, Node]   = {}   # node_id -> Node
        self.bridges    : Dict[Tuple[int,int], Bridge] = {}

        # inst_id -> list[nest_node_id]  (all sense nests for that instance)
        self.inst_to_nests : Dict[str, List[int]] = {}
        # word_node_id -> list[nest_node_id]
        self.word_to_nests : Dict[int, List[int]] = {}

        # All nest node_ids
        self.nest_ids      : List[int] = []

    # ── node registry ────────────────────────────────────────────────────────

    def _add_node(self, node: Node) -> Node:
        self.nodes[node.node_id] = node
        return node

    def _link(self, a: Node, b: Node, pheromone: float = 0.0):
        a.edges[b.node_id] = pheromone
        b.edges[a.node_id] = pheromone

    # ── graph construction ────────────────────────────────────────────────────

    def build_from_document(self, doc: SemEvalDocument):
        """
        Construct the full graph for one document.
        Plain nodes: text, sentence_i, word_j
        Nest  nodes: one per (inst_id, synset)
        """
        text_node = self._add_node(
            Node("text", f"text:{doc.doc_id}", energy=self.E0,
                 odour_length=self.odour_length))

        for s_idx, sentence in enumerate(doc.sentences):
            sent_node = self._add_node(
                Node("sentence", f"sent:{doc.doc_id}.s{s_idx:03d}",
                     energy=self.E0, odour_length=self.odour_length))
            self._link(text_node, sent_node)

            for (surface, inst_id) in sentence:
                if inst_id is None:
                    continue
                inst = doc.instances.get(inst_id)
                if inst is None:
                    continue

                word_node = self._add_node(
                    Node("word", f"word:{inst_id}",
                         energy=self.E0, odour_length=self.odour_length))
                self._link(sent_node, word_node)

                # Retrieve candidate synsets from WordNet
                synsets = wn.synsets(inst.lemma, pos=inst.pos)
                if not synsets:
                    synsets = wn.synsets(inst.lemma)  # pos-fallback
                if not synsets:
                    continue

                nests_for_inst: List[int] = []
                for syn in synsets:
                    nest = self._add_node(
                        Node("nest", f"nest:{inst_id}:{syn.name()}",
                             energy=self.E0,
                             odour_length=self.odour_length))
                    nest.synset    = syn
                    nest.word_lemma = inst.lemma
                    nest.word_pos   = inst.pos
                    nest.inst_id    = inst_id
                    # Nest odour = indexed definition vector (fixed)
                    nest.odour_vector = set(self.lesk.synset_vector(syn))
                    self._link(word_node, nest)
                    self.nest_ids.append(nest.node_id)
                    nests_for_inst.append(nest.node_id)

                self.inst_to_nests[inst_id]     = nests_for_inst
                self.word_to_nests[word_node.node_id] = nests_for_inst

    # ── bridge management ─────────────────────────────────────────────────────

    def _bridge_key(self, id_a: int, id_b: int) -> Tuple[int, int]:
        return (min(id_a, id_b), max(id_a, id_b))

    def add_bridge(self, nest_a: Node, nest_b: Node,
                   pheromone: float = 0.01):
        """Create a bridge between two potential-friend nest nodes."""
        key = self._bridge_key(nest_a.node_id, nest_b.node_id)
        if key not in self.bridges:
            bridge = Bridge(nest_a, nest_b, pheromone)
            self.bridges[key] = bridge
            # Register as a regular edge so movement probability uses it
            nest_a.edges[nest_b.node_id] = pheromone
            nest_b.edges[nest_a.node_id] = pheromone

    def remove_dead_bridges(self):
        """Delete bridges whose pheromone dropped to 0."""
        dead_keys = [k for k, b in self.bridges.items() if not b.alive]
        for key in dead_keys:
            b = self.bridges.pop(key)
            # Remove the edge entries from both nests
            n_a = self.nodes.get(b.nest_a_id)
            n_b = self.nodes.get(b.nest_b_id)
            if n_a: n_a.edges.pop(b.nest_b_id, None)
            if n_b: n_b.edges.pop(b.nest_a_id, None)

    def evaporate_bridges(self, delta: float):
        for bridge in self.bridges.values():
            bridge.evaporate(delta)

    def deposit_bridge_pheromone(self, nest_a_id: int, nest_b_id: int,
                                  amount: float):
        key = self._bridge_key(nest_a_id, nest_b_id)
        if key in self.bridges:
            self.bridges[key].deposit(amount)
            # Keep edge pheromone in sync
            na = self.nodes.get(nest_a_id)
            nb = self.nodes.get(nest_b_id)
            if na: na.edges[nest_b_id] = self.bridges[key].pheromone
            if nb: nb.edges[nest_a_id] = self.bridges[key].pheromone

    def is_bridge(self, id_a: int, id_b: int) -> bool:
        return self._bridge_key(id_a, id_b) in self.bridges

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def neighbours(self, node: Node) -> List[Node]:
        return [self.nodes[nid] for nid in node.edges if nid in self.nodes]

    def enemy_nest_ids(self, nest: Node) -> Set[int]:
        """All nests that share the same inst_id (different sense, same word)."""
        return set(self.inst_to_nests.get(nest.inst_id, [])) - {nest.node_id}

    def is_potential_friend(self, home_nest: Node, other_nest: Node) -> bool:
        """
        A nest is a 'potential friend' if it corresponds to a *different* word
        (different inst_id) and is not the home nest.
        """
        return (other_nest.is_nest
                and other_nest.inst_id != home_nest.inst_id
                and other_nest.node_id != home_nest.node_id)


# ─────────────────────────────────────────────
#  SECTION 6 – ACA PARAMETERS
# ─────────────────────────────────────────────

class ACAParams:
    """All tunable parameters of the Ant Colony Algorithm."""
    def __init__(self):
        self.E_a          : float = 5.0    # Energy taken per node visit
        self.E_max        : float = 30.0   # Max energy an ant can carry
        self.delta        : float = 0.1    # Pheromone evaporation rate
        self.E0           : float = 20.0   # Initial energy per node
        self.omega        : int   = 15     # Ant lifespan (cycles)
        self.LV           : int   = 100    # Odour vector length
        self.delta_V      : float = 20.0   # % of odour deposited per visit
        self.n_cycles     : int   = 100    # Total simulation cycles
        self.theta        : float = 1.0    # Pheromone deposited per edge crossing
        self.bridge_init_pheromone: float = 0.5


# ─────────────────────────────────────────────
#  SECTION 7 – MOVEMENT PROBABILITIES
# ─────────────────────────────────────────────

def eval_node_explore(node: Node, all_neighbours: List[Node]) -> float:
    """
    Exploration mode: attracted to high-energy nodes.
    Eval_f(N) = E(N) / sum_k E(N_k)
    """
    total = sum(n.energy for n in all_neighbours)
    if total == 0:
        return 1.0 / max(len(all_neighbours), 1)
    return node.energy / total


def eval_edge_explore(pheromone: float) -> float:
    """
    Exploration mode: *avoid* pheromone-heavy edges.
    Eval_f(A) = 1 - phi(A)    (normalised to [0,1])
    """
    return max(0.0, 1.0 - pheromone)


def eval_node_return(node: Node, ant: Ant, lesk: LeskMeasure,
                     all_neighbours: List[Node]) -> float:
    """
    Return mode: attracted to nodes whose odour resembles the ant's own.
    Eval_f(N) = ExtLesk(V(N), V(f_A)) / sum_k ExtLesk(V(N_k), V(f_A))
    """
    scores = []
    ant_v  = ant.odour_vector
    for n in all_neighbours:
        s = lesk.score_two_vectors(frozenset(n.odour_vector), ant_v)
        scores.append(s)
    total = sum(scores)
    if total == 0:
        return 1.0 / max(len(all_neighbours), 1)
    idx = all_neighbours.index(node)
    return scores[idx] / total


def eval_edge_return(pheromone: float) -> float:
    """Return mode: *follow* pheromone-heavy edges. Eval_f(A) = phi(A)."""
    return pheromone


def compute_transition_probabilities(ant: Ant, current_node: Node,
                                     graph: ACAGraph,
                                     lesk: LeskMeasure,
                                     params: ACAParams) -> List[Tuple[Node, float]]:
    """
    Compute P(N_i, A_j) for each neighbour of current_node.

    Returns list of (neighbour_node, probability).

    Special case – potential friend nests adjacent to current_node:
    when current_node is adjacent to a friend nest we include that nest in the
    candidate list with Eval_f(A) = 0 (pheromone ignored on that edge, as per
    the paper).
    """
    neighbours = graph.neighbours(current_node)
    if not neighbours:
        return []

    # Identify enemy nests (same word as home) – ants never voluntarily enter
    enemy_ids = graph.enemy_nest_ids(ant.mother_nest)

    # Filter out enemies
    candidates = [n for n in neighbours if n.node_id not in enemy_ids]
    if not candidates:
        candidates = neighbours  # fallback: allow all

    weights = []
    for nb in candidates:
        pheromone = current_node.edges.get(nb.node_id, 0.0)

        # Is this a bridge edge?
        on_bridge = graph.is_bridge(current_node.node_id, nb.node_id)

        if ant.mode == "explore":
            e_node = eval_node_explore(nb, candidates)
            # If nb is a potential friend nest: pheromone ignored (Eval_f(A)=0)
            if nb.is_nest and graph.is_potential_friend(ant.mother_nest, nb):
                e_edge = 0.0
            else:
                e_edge = eval_edge_explore(pheromone) if not on_bridge else pheromone
            w = e_node + e_edge
        else:  # return
            e_node = eval_node_return(nb, ant, lesk, candidates)
            if nb.is_nest and graph.is_potential_friend(ant.mother_nest, nb):
                e_edge = 0.0
            else:
                e_edge = eval_edge_return(pheromone)
            w = e_node + e_edge

        weights.append(max(w, 1e-9))

    total = sum(weights)
    probs = [(nb, w / total) for nb, w in zip(candidates, weights)]
    return probs


def choose_next_node(probs: List[Tuple[Node, float]]) -> Optional[Node]:
    """Stochastic selection according to transition probabilities."""
    if not probs:
        return None
    nodes, weights = zip(*probs)
    return random.choices(nodes, weights=weights, k=1)[0]


# ─────────────────────────────────────────────
#  SECTION 8 – ANT PRODUCTION
# ─────────────────────────────────────────────

def nest_production_probability(nest: Node) -> float:
    """
    Sigmoid production probability from the paper:
    P(N_A) = arctan(E(N)) / pi + 0.5
    """
    return math.atan(nest.energy) / math.pi + 0.5


def maybe_produce_ant(nest: Node, params: ACAParams,
                       lesk: LeskMeasure) -> Optional[Ant]:
    """
    At the start of each cycle, a nest may produce one ant, using 1 energy unit.
    """
    if nest.energy < 1.0:
        return None
    p = nest_production_probability(nest)
    if random.random() < p:
        nest.energy -= 1.0
        odour = frozenset(lesk.synset_vector(nest.synset)) if nest.synset else frozenset()
        ant = Ant(mother_nest=nest,
                  lifespan=params.omega,
                  e_max=params.E_max,
                  odour_vector=odour)
        return ant
    return None


def build_lemma_mapping(documents) -> Dict[str, str]:
    """
    Construieste mapping BN->WN folosind lema si POS din corpus.
    Nu e perfect dar e functional.
    """
    pos_map = {"n": wn.NOUN, "v": wn.VERB, "r": wn.ADV, "a": wn.ADJ}
    mapping = {}
    not_found = []

    for doc in documents:
        for inst_id, inst in doc.instances.items():
            for key in inst.gold_senses:
                key = key.strip()
                if not key.startswith("bn:") or key in mapping:
                    continue
                m = re.match(r"bn:(\d+)([nvra])$", key)
                if not m:
                    continue
                pos_char = m.group(2)
                pos = pos_map.get(pos_char)
                if pos is None:
                    continue

                synsets = wn.synsets(inst.lemma, pos=pos)
                if not synsets:
                    synsets = wn.synsets(inst.lemma)

                if synsets:
                    mapping[key] = synsets[0].name()
                else:
                    not_found.append((key, inst.lemma))

    print(f"[Mapping] {len(mapping)} mapped, {len(not_found)} not found")
    return mapping

# ─────────────────────────────────────────────
#  SECTION 9 – MAIN SIMULATION CYCLE
# ─────────────────────────────────────────────




class ACASimulation:
    """
    Orchestrates the Ant Colony Algorithm simulation for one document.
    """

    def __init__(self, graph: ACAGraph, params: ACAParams, lesk: LeskMeasure):
        self.graph   = graph
        self.params  = params
        self.lesk    = lesk
        self.ants    : List[Ant] = []
        self.cycle   : int       = 0

        # Majority vote: nest_node_id -> how many times it was 'selected'
        # across all cycles (energy returned to nest contributes)
        self.majority_votes: Dict[int, int] = defaultdict(int)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _all_nests(self) -> List[Node]:
        return [self.graph.nodes[nid] for nid in self.graph.nest_ids
                if nid in self.graph.nodes]

    # ── per-cycle steps ───────────────────────────────────────────────────────

    def _step1_cleanup(self):
        """
        Step 1: Remove dead ants; dead ant deposits its energy on current node.
        Remove bridges with zero pheromone.
        """
        surviving = []
        for ant in self.ants:
            if not ant.alive:
                # Deposit carried energy on the node where it died
                ant.current_node.energy += ant.energy
            else:
                surviving.append(ant)
        self.ants = surviving
        self.graph.remove_dead_bridges()

    def _step2_produce_ants(self):
        """
        Step 2: Each nest may produce one ant per cycle.
        There is no explicit maximum number of ants per nest; production is
        probabilistic (sigmoid). In practice energy limits production naturally.
        """
        for nest in self._all_nests():
            ant = maybe_produce_ant(nest, self.params, self.lesk)
            if ant is not None:
                self.ants.append(ant)
                # Register ant at the nest node
                nest.visiting_ants.append(ant)

    def _step3_move_ants(self):
        """
        Step 3: For each ant decide mode, move, possibly create bridge.
        """
        for ant in self.ants:
            if not ant.alive:
                continue

            # Clear visiting list from previous cycle
            ant.current_node.visiting_ants = \
                [a for a in ant.current_node.visiting_ants if a.ant_id != ant.ant_id]

            # ── decide mode ──────────────────────────────────────────────────
            ant.maybe_switch_to_return()

            # ── compute movement probs ───────────────────────────────────────
            probs = compute_transition_probabilities(
                ant, ant.current_node, self.graph, self.lesk, self.params)

            if not probs:
                # Nowhere to go → stay
                next_node = ant.current_node
            else:
                next_node = choose_next_node(probs)

            # ── handle bridge creation ────────────────────────────────────────
            # If next_node is a *potential friend* nest → build bridge and
            # switch to return mode (ant goes home via bridge).
            if (next_node.is_nest
                    and self.graph.is_potential_friend(ant.mother_nest, next_node)
                    and ant.mode == "explore"):
                # Build bridge between mother nest and this nest
                self.graph.add_bridge(ant.mother_nest, next_node,
                                      self.params.bridge_init_pheromone)
                # Ant crosses to friend nest, deposits pheromone, then goes home
                self._cross_edge(ant, ant.current_node, next_node)
                ant.mode = "return"
                # Now move ant toward home via bridge (next step will handle)
                ant.current_node = next_node

            else:
                # Normal move
                self._cross_edge(ant, ant.current_node, next_node)
                ant.current_node = next_node

            # ── arrive at mother nest in return mode ──────────────────────────
            if (ant.mode == "return"
                    and ant.current_node.node_id == ant.mother_nest.node_id):
                # Deposit energy to nest
                ant.mother_nest.energy += ant.energy
                # Majority vote: this nest is reinforced
                self.majority_votes[ant.mother_nest.node_id] += 1
                ant.mother_nest.vote_count += 1
                ant.energy = 0.0
                ant.mode   = "explore"

            # ── collect energy (explore mode) ─────────────────────────────────
            if ant.mode == "explore" and not ant.current_node.is_nest:
                ant.pick_up_energy(ant.current_node, self.params.E_a)

            # Register ant visit
            ant.current_node.visiting_ants.append(ant)
            ant.path.append(ant.current_node.node_id)

    def _cross_edge(self, ant: Ant, from_node: Node, to_node: Node):
        """
        Deposit pheromone on the crossed edge.
        phi_{t+1}(A) = phi_t(A) + theta
        Also deposit odour on the destination plain node.
        """
        # Pheromone
        eid = to_node.node_id
        from_node.edges[eid]       = from_node.edges.get(eid, 0.0) + self.params.theta
        to_node.edges[from_node.node_id] = to_node.edges.get(from_node.node_id, 0.0) + self.params.theta

        # If crossing a bridge, update bridge pheromone too
        if self.graph.is_bridge(from_node.node_id, to_node.node_id):
            self.graph.deposit_bridge_pheromone(from_node.node_id,
                                                 to_node.node_id,
                                                 self.params.theta)

        # Odour deposit on destination (plain nodes only)
        if not to_node.is_nest:
            to_node.deposit_odour(ant.odour_vector, self.params.delta_V)

    def _step4_update_environment(self):
        """
        Step 4: Evaporate pheromone on all edges and bridges.
        phi_{t+1}(A) = phi_t(A) * (1 - delta)
        """
        for node in self.graph.nodes.values():
            for nb_id in list(node.edges.keys()):
                old_phi = node.edges[nb_id]
                new_phi = old_phi * (1.0 - self.params.delta)
                node.edges[nb_id] = max(0.0, new_phi)

        # Bridge-specific evaporation (may collapse bridge)
        self.graph.evaporate_bridges(self.params.delta)
        self.graph.remove_dead_bridges()

        # Tick ant lifespans
        for ant in self.ants:
            ant.tick()

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True):
        """Execute the full simulation for n_cycles cycles."""
        for c in range(self.params.n_cycles):
            self.cycle = c + 1

            self._step1_cleanup()
            self._step2_produce_ants()
            self._step3_move_ants()
            self._step4_update_environment()

            if verbose and (c % 10 == 0 or c == self.params.n_cycles - 1):
                n_alive = sum(1 for a in self.ants if a.alive)
                n_bridges = len(self.graph.bridges)
                print(f"  Cycle {self.cycle:4d} | ants={n_alive:4d} "
                      f"| bridges={n_bridges:4d}")

    # ─────────────────────────────────────────────
    #  SECTION 10 – SENSE SELECTION & OUTPUT
    # ─────────────────────────────────────────────

    def select_senses(self, strategy: str = "energy") -> Dict[str, str]:
        """
        Determine the winning sense for each word instance.

        strategy='energy'  : sense whose nest has the most energy
        strategy='vote'    : sense whose nest received the most votes (majority vote)

        Returns: inst_id -> synset.name()
        """
        result: Dict[str, str] = {}

        for inst_id, nest_ids in self.graph.inst_to_nests.items():
            if not nest_ids:
                continue
            nests = [self.graph.nodes[nid] for nid in nest_ids
                     if nid in self.graph.nodes]
            if not nests:
                continue

            if strategy == "vote":
                best = max(nests, key=lambda n: self.majority_votes.get(n.node_id, 0))
            else:
                best = max(nests, key=lambda n: n.energy)

            if best.synset:
                result[inst_id] = best.synset.name()

        return result

    def print_node_visitors(self, node_id: int):
        """Diagnostic: print which ants are currently visiting a node."""
        node = self.graph.get_node(node_id)
        if node is None:
            print(f"[Node {node_id}] Not found.")
            return
        print(f"[Node {node_id} – {node.label}]  "
              f"Visiting ants this cycle: {len(node.visiting_ants)}")
        for ant in node.visiting_ants:
            print(f"   {ant}")


# ─────────────────────────────────────────────
#  SECTION 11 – EVALUATION METRICS
# ─────────────────────────────────────────────

def sense_key_to_synset_name(sense_key: str) -> Optional[str]:
    """
    Convert a WordNet sense key  (e.g. 'bank%1:14:01::')
    to a synset name (e.g. 'bank.n.01').
    Returns None if conversion fails.
    """
    try:
        lemma = wn.lemma_from_key(sense_key)
        return lemma.synset().name()
    except Exception:
        return None


_BN_TO_WN: Dict[str, str] = {}  # populat in main()

def coarse_grain_match(predicted_synset_name: str,
                       gold_sense_keys: List[str]) -> bool:
    for key in gold_sense_keys:
        key = key.strip()
        if key.startswith("bn:"):
            gold_name = _BN_TO_WN.get(key)
            if gold_name and gold_name == predicted_synset_name:
                return True
        else:
            gold_name = sense_key_to_synset_name(key)
            if gold_name and gold_name == predicted_synset_name:
                return True
    return False


def evaluate(predictions: Dict[str, str],
             documents: List[SemEvalDocument]) -> Dict[str, float]:
    """
    Compute precision, recall and F1 (micro-averaged) as used in SemEval.

    precision = #correct_answered / #answered
    recall    = #correct_answered / #total_instances
    F1        = 2 * P * R / (P + R)
    """
    total    = 0
    answered = 0
    correct  = 0

    for doc in documents:
        for inst_id, inst in doc.instances.items():
            total += 1
            pred = predictions.get(inst_id)
            if pred is None:
                continue
            answered += 1
            if coarse_grain_match(pred, inst.gold_senses):
                correct += 1

    precision = correct / answered if answered else 0.0
    recall    = correct / total    if total    else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall"   : recall,
        "F1"       : f1,
        "correct"  : correct,
        "answered" : answered,
        "total"    : total,
    }


def print_results(predictions: Dict[str, str],
                  documents: List[SemEvalDocument],
                  strategy: str):
    print(f"\n{'='*60}")
    print(f"  Sense selection strategy: {strategy}")
    print(f"{'='*60}")

    metrics = evaluate(predictions, documents)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['F1']:.4f}")
    print(f"  Correct   : {metrics['correct']} / {metrics['answered']} answered"
          f" / {metrics['total']} total")

    print(f"\n  Sample predictions:")
    for inst_id, pred in list(predictions.items())[:10]:
        for doc in documents:
            inst = doc.instances.get(inst_id)
            if inst:
                match = coarse_grain_match(pred, inst.gold_senses)
                mark  = "✓" if match else "✗"
                print(f"    [{mark}] {inst_id:30s}  "
                      f"pred={pred:30s}  gold={inst.gold_senses}")
                break

    return metrics


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ant Colony Algorithm for Word Sense Disambiguation")
    print("  (Schwab et al., COLING 2012)")
    print("=" * 60)

    # ── Parametri ────────────────────────────────────────────────
    params = ACAParams()
    params.E_a                   = 5.0
    params.E_max                 = 30.0
    params.delta                 = 0.1
    params.E0                    = 20.0
    params.omega                 = 15
    params.LV                    = 100
    params.delta_V               = 20.0
    params.n_cycles              = 30
    params.theta                 = 1.0
    params.bridge_init_pheromone = 0.05

    # ── Corpus ───────────────────────────────────────────────────
    print("\n[1] Loading corpus …")

    xml_file = os.path.join("evaluation_datasets", "test-en-coarse",
                            "test-en-coarse.data.xml")
    key_file = os.path.join("evaluation_datasets", "test-en-coarse",
                            "test-en-coarse.gold.key.txt")

    documents = load_semeval2007_task7(xml_file, key_file)

    # ADAUGA IMEDIAT DUPA:
    global _BN_TO_WN
    _BN_TO_WN = build_lemma_mapping(documents)

    if not documents:
        print("[ERROR] No documents loaded.")
        return

    total_instances = sum(len(d.instances) for d in documents)
    print(f"\n[TOTAL] {len(documents)} documents, {total_instances} instances.")

    # ── Lesk ─────────────────────────────────────────────────────
    print("\n[2] Initialising Extended Lesk …")
    lesk = LeskMeasure()

    # ── Rulare per document ───────────────────────────────────────
    all_predictions_energy = {}
    all_predictions_vote   = {}

    for doc_idx, doc in enumerate(documents):
        if not doc.instances:
            continue

        print(f"\n[Doc {doc_idx+1}/{len(documents)}]  "
              f"id={doc.doc_id}  instances={len(doc.instances)}")

        graph = ACAGraph(lesk=lesk, E0=params.E0, odour_length=params.LV)
        graph.build_from_document(doc)

        if not graph.nest_ids:
            print("  No candidate senses. Skipping.")
            continue

        print(f"  Graph: nodes={len(graph.nodes)}  "
              f"nests={len(graph.nest_ids)}")

        sim = ACASimulation(graph, params, lesk)
        sim.run(verbose=True)

        all_predictions_energy.update(sim.select_senses("energy"))
        all_predictions_vote.update(sim.select_senses("vote"))

        # Diagnostic: cati furnici sunt pe primul nest
        sim.print_node_visitors(graph.nest_ids[0])

    # ── Evaluare ──────────────────────────────────────────────────
    global BN_TO_WN
    BN_TO_WN = load_babelnet_to_wordnet_mapping("bn_wn_mapping.tsv")
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    print_results(all_predictions_energy, documents, "energy")
    print_results(all_predictions_vote,   documents, "majority-vote")


if __name__ == "__main__":
    random.seed(42)
    main()