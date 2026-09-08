"""Curated CRW2 BPSEQ fixtures used in the published COIN/SARNA studies."""
from __future__ import annotations
from dataclasses import dataclass
from importlib.resources import files

@dataclass(frozen=True,slots=True)
class RnaBenchmark:
    id:str; accession:str; organism:str; rna_class:str; sequence:str; dot_bracket:str; pair_count:int; source_url:str; paper_group:str; size:str

SPECS={
 "X67579":("Saccharomyces cerevisiae","5S rRNA","https://crw2-comparative-rna-web.org/wp-content/uploads/2022/10/d.5.e.S.cerevisiae.bpseq_-3.txt","COIN 2017 + SARNA-Predict","medium"),
 "AF034620":("Haloarcula marismortui","5S rRNA","https://crw2-comparative-rna-web.org/wp-content/uploads/2022/10/d.5.a.H.marismortui.bpseq_-3.txt","COIN 2017 + SARNA-Predict","medium"),
 "X05914":("Drosophila virilis","mitochondrial 16S rRNA","https://crw2-comparative-rna-web.org/wp-content/uploads/2022/11/d.16.m.D.virilis.bpseq_.txt","COIN 2017 + SARNA-Predict","large"),
}

ARCHIVE_SPECS={
 "AR2-SRP-28":("Shigella flexneri","SRP RNA","tiny"),
 "AR2-SRP-30":("Bacillus amyloliquefaciens","SRP RNA","tiny"),
 "AR2-SRP-33":("Bradyrhizobium japonicum","SRP RNA","tiny"),
 "AR2-TRNA-54":("Ascaris suum","tRNA Arg","easy"),
 "AR2-TRNA-58":("Ascaris suum","tRNA Gln","easy"),
 "AR2-TRNA-62":("Ascaris suum","tRNA Phe","easy"),
 "AR2-TRNA-70":("Bos taurus","tRNA Phe","teaching"),
 "AR2-TRNA-72":("Homo sapiens","tRNA Ile","teaching"),
 "AR2-TRNA-76":("Mycoplasma capricolum","tRNA Ala","teaching"),
 "AR2-TRNA-90":("Escherichia coli","tRNA Ser","medium"),
 "AR2-RNASEP-300":("ArchiveII SM-A1519","RNase P RNA","large"),
 "AR2-SRP-300":("Branchiostoma floridae","SRP RNA","large"),
}

ARCHIVE_DATA={
 "AR2-SRP-28":('CCGUCAGGUCCGGAAGGAAGCAGCGGUA','((((....(((....)))....))))..',7),
 "AR2-SRP-30":('AACCAUGUCAGGUCCGGAAGGAAGCAGCAU','....((((....(((....)))....))))',7),
 "AR2-SRP-33":('AACCGGGUCAGGUCCGGAAGGAAGCAGCCCUAA','....((((....(((....)))....))))...',7),
 "AR2-TRNA-54":('GACAAAUGUUUUCAGGUCUUCUAAAUCUGUUUUGGAGAAAUCCGUUUGUUUCCA','(((((((............((((((.....)))))).......)))))))....',13),
 "AR2-TRNA-58":('UAUACUUUAGUUUAGGAAGAAUAUUUAUUUUUGGUGUAAAAGGGUUGUAGUAUAGCCA','((((((...((((.....)))).(((((.......)))))........))))))....',15),
 "AR2-TRNA-62":('ACUCUGUUAGUUUAUGUUUUAAAAUAUGACUUUGAAGAAGUUGGAAAAUGUUAGGAGUGCCA','(((((....((((........)))).(((((.......)))))....(..)..)))))....',15),
 "AR2-TRNA-70":('GUUGAUGUAGCUUAACCCAAAGCAAGGCACUGAAAAUGCCUAGAUGAGUCUCCCAACUCCAUAAACACCA','(((.(((..((((......)))).(((((.......)))))....((((......))))))).)))....',19),
 "AR2-TRNA-72":('AGAAAUAUGUCUGAUAAAAGAGUUACUUUGAUAGAGUAAAUAAUAGGAGCUUAAACCCCCUUAUUUCUACCA','(((((((..(((......))).(((((.......))))).....(((.(.......).))))))))))....',19),
 "AR2-TRNA-76":('GGGCCCUUAGCUCAGCUGGGAGAGCACCUGCCUUGCACGCAGGGGGUCGACGGUUCGAUCCCGUUAGGGUCCACCA','(((((((..((((........)))).(((((.......))))).....(((((.......))))))))))))....',21),
 "AR2-TRNA-90":('GGAGAGAUGCCGGAGCGGCUGAACGGACCGGUCUCGAAAACCGGAGUAGGGGCAACUCUACCGGGGGUUCAAAUCCCCCUCUCUCCGCCA','(((((((..(((...........))).(((((.......)))))..................(((((.......))))))))))))....',20),
 "AR2-RNASEP-300":('GAGGAAAGUCCGGGCUCCACAGGGCAGGGUGCCAGGUAAUCCCUGGGGGGCGCGAGCCUACGGAAAGUGCAACAGAAAAUAUACCGCCAGGAAACUGGUAAGGGUGAAAUGGUGCGGUAAGAGCGCACCGCGCAGGUGGUAACACGCUGCGGCAUGGUAAACCCCACCCGGAGCAAGACCAAAUAGGAAAGCGAUGCCGUGGCCCGCGGCGCUUUCGGGUAGGUCGCACGAGGCCGCCGGUGACGGCGGUCCCAGAUGAAUGAUUGCCACAGAUUAACUUCUGGACAGAACCCGGCUUAU','.....(((.(((((((((...((((.((((((((((.....)))))(((((....)))).)((...((((.............(((((((....)))))..)).......((((((.......))))))(((((((((....))).))))).)...))..)))))))))))))...((((......((((((...(((((.))))))))))))))).....))))......((((((((....))))))))..................((((......)))).......))))))))..',86),
 "AR2-SRP-300":('GCUGGGCGUGGUGGCGCGCGCCUGUAAUCCAGCUACCUGGAGGCUGGGGUUGGUGGACCGCUUGAGGCCGGGAGUUCUGCGGCGUGCUGUCCUAUGGCGAUCGGGCGUCCACGCUAAGUUCGGCAUCGAUAUGGUUACCCCGGGGGAGCCCAGGGCAACCAGGUCGUCUAAGGAGGGACGCACCGGCCCAGGUUGGAAACAAAGCAGGCAAAAGUCCCCGUGUCGAUCAGUAGCGGGAUCGCGCCCGUGAAUAGGCACUGCGUUUCAGCCCGGCCAAUAUAGUGGGACCCAACUCUUUUU','((((((((((......)))))))....(((........)))))).((((((((....(((((...(((((((....(((.((((((.((.(((((......(((((((((.(((((.(.(((((((((((.(((((.(((.(((....))).))).))))).))))((...)).(((((......(((....(((....)))....)))....))))).))))))).)..)))))))...)))))))...))))))).)))))).)))))))))).....)))))...))))))))....',99),
}

def _parse(accession:str)->RnaBenchmark:
    if accession in ARCHIVE_SPECS:
        organism,rna_class,size=ARCHIVE_SPECS[accession]
        sequence,structure,pairs=ARCHIVE_DATA[accession]
        return RnaBenchmark(accession,accession,organism,rna_class,sequence,structure,pairs,
            "https://github.com/mxfold/mxfold2/releases/download/v0.1.1/archiveII.tar.gz",
            "ArchiveII · published known structure",size)
    organism,rna_class,url,paper_group,size=SPECS[accession]
    text=files("coin.problems.rna_benchmarks").joinpath(f"{accession}.bpseq").read_text(encoding="utf-8")
    rows=[]
    for line in text.splitlines():
        fields=line.split()
        if len(fields)==3 and fields[0].isdigit() and fields[1].upper() in "ACGUT": rows.append((int(fields[0]),fields[1].upper().replace("T","U"),int(fields[2])))
    if not rows or [x[0] for x in rows]!=list(range(1,len(rows)+1)): raise ValueError(f"invalid BPSEQ fixture: {accession}")
    structure=["."]*len(rows); pairs=0
    for index,_,partner in rows:
        if partner>index: structure[index-1]="("; structure[partner-1]=")"; pairs+=1
    return RnaBenchmark(accession,accession,organism,rna_class,"".join(x[1] for x in rows),"".join(structure),pairs,url,paper_group,size)

def list_benchmarks()->list[RnaBenchmark]:
    order=("AR2-SRP-28","AR2-SRP-30","AR2-SRP-33","AR2-TRNA-54","AR2-TRNA-58","AR2-TRNA-62",
           "AR2-TRNA-70","AR2-TRNA-72","AR2-TRNA-76","AR2-TRNA-90","X67579","AF034620",
           "AR2-RNASEP-300","AR2-SRP-300","X05914")
    return [_parse(x) for x in order]
def get_benchmark(identifier:str)->RnaBenchmark:
    key=identifier.upper()
    if key not in SPECS and key not in ARCHIVE_SPECS: raise KeyError(identifier)
    return _parse(key)
