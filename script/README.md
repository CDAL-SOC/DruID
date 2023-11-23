Support only hg38 genome.

## Packages and databases required
1. transvar (required python2.7) https://transvar.readthedocs.io/en/latest/download_and_install.html
transvar db required:

`transvar config --download_anno --refversion hg38`

Note that you'll also need to download the reference files https://storage.googleapis.com/genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.fasta and https://storage.googleapis.com/genomics-public-data/resources/broad/hg38/v0/Homo_sapiens_assembly38.fasta.fai before you run ./goAAtoGv2.sh

Annovar https://annovar.openbioinformatics.org/en/latest/
This tool requires registration using https://www.openbioinformatics.org/annovar/annovar_download_form.php - post successful registration you'd be receiving an email with the link to download the tool.

2. Annovar db required: i) refgene ii) dbnsfp42c

You can use the below command to download the same:

`perl annovar/annotate_variation.pl -downdb -webfrom annovar refGene humandb/ --buildver hg38 perl annovar/annotate_variation.pl -downdb -webfrom annovar dbnsfp42c humandb/ --buildver hg38`

Related files will be downloaded to the folder named "humandb/". Also, update the relevant variables (BASE_PATH, REFERENCE_FILE_PATH) in goAAtoGv*.sh as required

To run

`bash ./goAAtoGv2.sh input_file`

input_file format: per row, gene:Amino_Acid_Change eg, TP53:E120K

output is the annovar annotation of functional prediction from various algorihtm prediction.