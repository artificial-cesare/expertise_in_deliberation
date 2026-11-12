"""Speaker roles in the UK parliamentary hearings, the expert reference corpus (2026-11-12).

Each hearing transcript holds three kinds of turns: the witnesses who give evidence, the committee members (MPs and
peers) who question them, and the chair. The witnesses below were labelled by hand from the "Witnesses" list at the
top of each hearing's HTML page (data/assemblies/raw/experts/); every other named speaker is a committee member.
Chairs and transcription placeholders (SPEAKER_NAME, SPEAKER 1, a stray FACILITATOR) get the role "chair".

expert_corpus() returns the substantial hearing utterances (>= 5 content words, non-empty text) in exactly the order
of the cached embeddings (outputs/paper_v2/embedding_check/{base,finetuned}_experts.npy, 1,667 rows), with columns
topic, file, hearing, speaker, role, duplicate. The expert reference of the embedding measures is
  witness & ~duplicate
(the gender hearing committees.parliament.uk is a byte-identical copy of committees.parliament.uk5).
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path("data/assemblies/processed_files/experts")
TOPICS = ("constitution", "future", "gender")
DUPLICATE = ("gender", "committees.parliament.uk_speeches.csv")  # copy of committees.parliament.uk5

# hearing (topic/file stem) -> witness speaker labels as they appear in the CSVs, spelling variants included
WITNESSES = {
    "constitution/3": ["NATASCHA ENGEL"],
    "constitution/4": ["KAY JENKINS"],
    "constitution/committees.parliament.uk1": ["PAT McFADDEN MP", "PAT Mc FADDEN MP", "PAT MC FADDEN MP",
                                               "NICK THOMAS-SYMONDS MP", "DARREN TIERNEY"],
    "constitution/committees.parliament.uk2": ["PROFESSOR LAURA MC ALLISTER", "DR ROWAN WILLIAMS"],
    "constitution/committees.parliament.uk": ["DR GILL", "ALAN COGBILL", "MARTIN WHEATLEY"],
    "future/committees.parliament.uk1": ["SALLY JONES", "JOHN COOKE", "NEIL ROSS"],
    "future/committees.parliament.uk2": ["SIR PETER SCHOFIELD", "JULIE TAYLOR", "PAUL MORRISON", "TOM GREIG"],
    "future/committees.parliament.uk3": ["GIDEON RABINOWITZ", "BETHAN LEWIS", "DR HAGEN-ZANKER", "DR JESSICA HAGEN-ZANKER",
                                         "DAPHNE JAYASINGHE", "DAPNE JAYASINGHE", "LOUIS HOFFMANN",
                                         "HÉLÈNE KUPERMAN-LE-BIHAN"],
    "future/committees.parliament.uk4": ["PAUL THOMAS", "HUGH ELLIS", "DR ELLIS", "TOM CHANCE"],
    "future/committees.parliament.uk": ["DAVID HENIG", "PROFESSOR USHERWOOD", "CATHERINE Mc BRIDE", "JOHN SPRINGFORD"],
    "gender/committees.parliament.uk2": ["LEWIS BROOKS", "NIC HAILEY", "EVA TABBASAM", "MIKE JOBBINS"],
    "gender/committees.parliament.uk3": ["DEBBIE WOSSKOW", "DR TREANOR", "STEPHEN WELTON", "JILL PAY", "DR MARKS",
                                         "PROFESSOR STEPHAN"],
    "gender/committees.parliament.uk4": ["DR CLARKSON", "PROFESSOR POPE", "DR NICHOLSON", "PROFESSOR PIELICHATY"],
    "gender/committees.parliament.uk5": ["CAT BURTON", "LUI ASQUITH", "NANCY KELLEY", "DR HAMLIN", "DR. HAMLIN"],
}
WITNESSES["gender/committees.parliament.uk"] = WITNESSES["gender/committees.parliament.uk5"]  # the duplicate
CHAIR = re.compile(r"^(CHAIR|CHAIRPERSON|SPEAKER_NAME|SPEAKER \d|FACILITATOR)$")


def hearing_role(hearing: str, speaker: str) -> str:
    if speaker in WITNESSES[hearing]: return "witness"
    if CHAIR.match(str(speaker).upper().strip()): return "chair"
    return "member"


def expert_corpus() -> pd.DataFrame:
    rows = []
    for topic in TOPICS:
        for f in sorted((ROOT / topic).glob("*_speeches.csv")):
            d = pd.read_csv(f)
            d = d[(d.num_non_function_words >= 5) & d.text.notna()]
            d = d[d.text.astype(str).str.strip() != ""]
            hearing = f"{topic}/{f.name.removesuffix('_speeches.csv')}"
            rows.append(d.assign(topic=topic, file=f.name, hearing=hearing, row=d.index,
                                 role=[hearing_role(hearing, s) for s in d.speaker],
                                 duplicate=(topic, f.name) == DUPLICATE))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    E = expert_corpus()
    print(len(E), "substantial hearing utterances (cache order)")
    print(E[~E.duplicate].groupby(["topic", "role"]).agg(turns=("speaker", "size"), speakers=("speaker", "nunique")).to_string())
