""" Utils for text_models  """

from typing import Any

import polars as pl
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from psycop.common.global_utils.paths import OVARTACI_SHARED_DIR
from psycop.common.global_utils.pickle import read_pickle, write_to_pickle


def chunk_dataframe(input_df: pl.DataFrame, n: int) -> list[pl.DataFrame]:
    chunk_size = -(-len(input_df) // n)  # Ceiling division to get chunk size
    return [
        input_df.slice(offset=i, length=chunk_size)
        for i in range(0, len(input_df), chunk_size)
    ]


def save_text_model_to_dir(
    model: Any,
    filename: str,
):
    """
    Saves the model to a pickle file

    Args:
        model (Any): The model to save
        filename (str): The filename to save the model as

    """
    filepath = OVARTACI_SHARED_DIR / "text_models" / filename
    write_to_pickle(model, filepath)


def load_text_model(
    filename: str,
) -> CountVectorizer | TfidfVectorizer:
    """
    Loads a text model from a pickle file

    Args:
        filename: filename name of the model
        path_str: path of model location
    """
    filepath = OVARTACI_SHARED_DIR / "text_models" / filename
    return read_pickle(filepath)


""" stop_words list from: Torp, Bertel. “Dansk Stopords Liste / Danish Stopwords.” Github, 6 Mar. 2020,
https://gist.github.com/berteltorp/0cf8a0c7afea7f25ed754f24cfc2467b. """

stop_words = """ ad
af
akkurat
al
aldrig
alene
alle
allerede
alligevel
alt
altid
altså
anden
andet
andre
art
at
bag
bare
begge
bla
blandt
blev
blive
bliver
blot
bringe
brug
burde
både
bør
ca
da
dag
de
del
dem
den
denne
dens
der
derefter
deres
derfor
derfra
deri
dermed
derpå
derved
des
desto
det
dets
dette
dig
din
dine
disse
dit
dog
du
dér
dét
efter
egen
ej
eller
ellers
en
end
endelig
endnu
ene
eneste
enhver
ens
enten
er
et
feks
faktisk
far
fat
fem
fik
find
finde
fire
flere
flest
fleste
for
foran
fordi
forrige
fortsat
fra
fx
få
får
følg
følge
følger
før
først
første
gang
gennem
gerne
gid
giv
giver
givet
gjorde
gjort
god
godt
gå
går
gør
gøre
gørende
ha
haft
ham
han
hans
har
havde
have
hej
hel
heldigvis
hele
heller
helst
helt
hen
hende
hendes
henover
her
herefter
heri
hermed
herpå
holder
hos
hun
hvad
hvem
hver
hvert
hvilke
hvilken
hvilkes
hvis
hvor
hvordan
hvorefter
hvorfor
hvorfra
hvorhen
hvori
hvorimod
hvornår
hvorved
hør
hører
hørt
hørte
i
igen
igennem
ikke
imellem
imens
imod
ind
indtil
ingen
intet
især
iøvrigt
ja
jeg
jer
jeres
jo
kan
kom
komme
kommer
kommet
kun
kunne
lad
langs
lav
lave
lavet
lidt
lig
lige
ligesom
ligeså
ligge
ligger
lille
lissom
længere
man
mand
mange
med
meget
mellem
men
mens
mere
mest
mig
min
mindre
mindst
mine
mit
mod
må
måske
måtte
måttet
ned
nej
nemlig
netop
ni
nogen
nogensinde
noget
nogle
nok
nu
ny
nye
nyt
nå
når
nær
næste
næsten
og
også
okay
om
omkring
op
os
otte
over
overalt
overhovedet
pga.
på
ret
rigt
ro
rundt
sagt
samme
sammen
se
seks
selv
selvom
senere
ser
ses
set
siden
sig
sige
siger
simpelthen
sin
sine
sit
skal
skam
ske
sker
skulle
små
småt
snart
som
stadig
stor
store
stort
synes
syntes
syv
så
sådan
således
såmænd
såvel
særlig
tag
tage
temmelig
thi
ti
tidligere
til
tilbage
tit
to
tre
tæt
ud
ude
uden
udover
under
undtagen
var
ved
vel
vi
via
videre
vil
ville
vis
vise
vist
vor
vore
vores
vort
vær
være
været
véd
år
års
én
ét
øvrigt""".split()
