import os

import numpy as np
import pandas as pd
import re
from gensim.models import KeyedVectors

# Modell betöltése
modellUtvonal = r'C:\Users\User\Downloads\Date-pairing/GoogleNews-vectors-negative300.bin.gz'
#modellUtvonal = r'D:\!SULI\13\AAF\dating/GoogleNews-vectors-negative300.bin.gz'
modell = KeyedVectors.load_word2vec_format(modellUtvonal, binary=True)

# Kulcsfogalmak
kulcsKifejezesek = [
    "szerelem", "gyûlölet", "boldogság", "szomorúság", "család", "barátok", "iskola", "stressz", "házi feladat", "kapcsolat",
    "álmok", "célok", "jövő", "emlékek", "megbánás", "düh", "félelem", "öröm", "konfliktus", "szorongás",
    "béke", "gyász", "unalom", "izgatottság", "csalódás", "remény", "változás", "önmagam", "barátság", "magány", "ihlet"
]

# Fájlok beolvasása

def fajlokBeolvasasa(mappa, eredmenyLista):
    for i in range(1, 20):
        fajlUt = os.path.join(mappa, f"{i}.txt")
        if os.path.exists(fajlUt):
            with open(fajlUt, "r", encoding="utf-8") as fajl:
                tartalom = fajl.read().strip()
                if tartalom:
                    eredmenyLista.append(tartalom)

def nagybetusSzavakSzama(szoveg):
    mondatok = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s', szoveg)
    szamlalo = 0
    for mondat in mondatok:
        szavak = mondat.split()
        if len(szavak) > 1:
            for szo in szavak[1:]:
                if szo.isupper():
                    szamlalo += 1
    return szamlalo

def mondatStrukturaBonyolultsag(szoveg):
    mondatok = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)\s(?=\.|\?|!|;)', szoveg)
    if not mondatok:
        return 0
    alarendeltSzamok = [
        len(re.split(r'(?<=\w),\s*(?=\w)|(?<=\w)\.\s*(?=\w)|(?<=\w)\s(and|or)\s(?=\w)', mondat))
        for mondat in mondatok
    ]
    return np.mean(alarendeltSzamok)


def kulcsKifejezesSuruseg(szoveg):
    szavak = szoveg.lower().split()
    szavakSzama = len(szavak)
    kulcsKifejezesDarabok = {kifejezes: szavak.count(kifejezes) for kifejezes in kulcsKifejezesek}
    osszKifejezes = sum(kulcsKifejezesDarabok.values())
    if szavakSzama == 0:
        return 0
    return osszKifejezes / szavakSzama

def temakPontozasa(tema, szoveg, modell, hasonlosagiKuszob=0.7):
    pont = 0
    szavak = re.findall(r'\b\w+\b', szoveg.lower())
    pont += szavak.count(tema.lower())
    for szo in szavak:
        if szo != tema.lower() and szo in modell and tema in modell:
            hasonlosag = modell.similarity(tema, szo)
            if hasonlosag >= hasonlosagiKuszob:
                pont += 1
    return pont

def jellemzokKinyerese(szoveg):
    atlagosSzoHossz = np.mean([len(szo) for szo in szoveg.split()])
    mondatokSzama = szoveg.count('.') + szoveg.count('!') + szoveg.count('?')
    Iskola = temakPontozasa("iskola", szoveg, modell)
    Munka = temakPontozasa("munka", szoveg, modell)
    szavak = szoveg.split()
    erzelmiIntenzitas = sum(1 for szo in szavak if szo.lower() in [
        "szerelem", "gyûlölet", "boldog", "szomorú", "csodálatos", "szörnyű"])/ len(szavak) if len(szavak) > 0 else 0
    suruseg = kulcsKifejezesSuruseg(szoveg)
    bonyolultsag = mondatStrukturaBonyolultsag(szoveg)
    felsorolasDb = szoveg.count(',')
    nagybetusDb = nagybetusSzavakSzama(szoveg)
    return [
        atlagosSzoHossz,
        mondatokSzama,
        Iskola,
        Munka,
        erzelmiIntenzitas,
        suruseg,
        bonyolultsag,
        felsorolasDb,
        nagybetusDb
    ]

def atlagSzamitas(df):
    fiuAtlagok,lanyAtlagok = [],[]
    for index, sor in df.iterrows():
        atlag = sor.mean()
        if index.startswith("FIU"):
            fiuAtlagok.append(atlag)
        elif index.startswith("LANY"):
            lanyAtlagok.append(atlag)
    return fiuAtlagok, lanyAtlagok

def legkozelebbiParokKivalasztasa(fiuAtlagok, lanyAtlagok):
    parok = []
    for lanyIndex, lanyPont in enumerate(lanyAtlagok):
        for fiuIndex, fiuPont in enumerate(fiuAtlagok):
            lanyId = f"Lány_{lanyIndex + 1}"
            fiuId = f"Fiú_{fiuIndex + 1}"
            kulonbseg = abs(lanyPont - fiuPont)
            parok.append((lanyId, fiuId, lanyPont, fiuPont, kulonbseg))
    parok.sort(key=lambda x: x[4])
    valasztottParok = []
    lanyok = set()
    fiuk = set()
    for lany, fiu, lanyPont, fiuPont, kulonbseg in parok:
        if lany not in lanyok and fiu not in fiuk:
            valasztottParok.append((lany, fiu, lanyPont, fiuPont, kulonbseg))
            lanyok.add(lany)
            fiuk.add(fiu)
        if len(valasztottParok) == 20:
            break
    for index, (lany, fiu, lanyPont, fiuPont, kulonbseg) in enumerate(valasztottParok):
        print(f"Pár {index + 1}: {lany}({lanyPont:.2f}) - {fiu}({fiuPont:.2f}) - eltérés: {kulonbseg:.2f}")

def main():
    fiuLista = []
    lanyLista = []
    fajlokBeolvasasa("male", fiuLista)
    fajlokBeolvasasa("female", lanyLista)
    osszesJellemzo = []
    azonositoLista = []
    for i, fiu in enumerate(fiuLista):
        jellemzok = jellemzokKinyerese(fiu)
        osszesJellemzo.append(jellemzok)
        azonositoLista.append(f"FIU{i + 1}")
    for i, lany in enumerate(lanyLista):
        jellemzok = jellemzokKinyerese(lany)
        osszesJellemzo.append(jellemzok)
        azonositoLista.append(f"LANY{i + 1}")
    oszlopok = [
        "Átlag szóhossz",
        "Mondatszám",
        "Iskola téma",
        "Munka téma",
        "Érzelmek vizsgálása",
        "Kulcskifejezések sûrűsége",
        "Bonyolultság",
        "Felsorolások száma",
        "Nagybetűs szavak száma"
    ]
    df = pd.DataFrame(osszesJellemzo, columns=oszlopok, index=azonositoLista)
    for oszlop in oszlopok:
        minErtek = df[oszlop].min()
        maxErtek = df[oszlop].max()
        df[oszlop] = df[oszlop].apply(lambda x: (x - minErtek) / (maxErtek - minErtek) * 99 + 1)
    fiuAtlagok, lanyAtlagok = atlagSzamitas(df)
    legkozelebbiParokKivalasztasa(fiuAtlagok, lanyAtlagok)

main()
