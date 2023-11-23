#have to switch java to ver 11 first
# sudo update-alternatives --config java

INPUTFILE=$1
BASE_PATH="/home/ajayago/annotation"
REFERENCE_FILE_PATH="/data/ajayago/druid/annotation/Homo_sapiens_assembly38.fasta"
DATA_PATH="/data/ajayago/druid/annotation/"

transvar panno --refversion hg38 --reference ${REFERENCE_FILE_PATH} --gencode -l ${INPUTFILE} -m 1 > ${INPUTFILE}.out.txt
#mutlist per role, TP53:E120K

cut -f 5 ${INPUTFILE}.out.txt | sed '1d' > ${INPUTFILE}.out2.txt

python ${BASE_PATH}/goAAtoG.py3 ${INPUTFILE}.out2.txt ${INPUTFILE} ${INPUTFILE}.out3.txt

perl ${BASE_PATH}/annovar/table_annovar.pl --thread 2 --buildver hg38 --outfile ${INPUTFILE}.annot --polish --remove --protocol dbnsfp42c,clinvar_20221231 --operation f,f --nastring '.' --otherinfo ${INPUTFILE}.out3.txt ${DATA_PATH}/humandb/

# need to check which column is the clinvar annotation inserted
cut -f 6-26,29,36-38,41-51,53-62,65-82,87-89,90,96-$NF ${INPUTFILE}.annot.hg38_multianno.txt > ${INPUTFILE}.annot.hg38_finalannot.txt
