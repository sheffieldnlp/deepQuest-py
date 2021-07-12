#! /bin/sh

#urls for data download
DATA_URLS="https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_mlqe.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_mlqe.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_mlqe.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_mlqe.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_mlqe.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_25k_wiki.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_100k_wiki.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_100k_wiki.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_100k_wiki.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_100k_wiki.tar.gz"

#urls for trained model downloads
MODELS_URL="https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_et_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ro_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_si_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_ne_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_mlqe_en_zh.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki25k_et_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ro_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_si_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_ne_en.tar.gz https://www.quest.dcs.shef.ac.uk/dq_student_birnn/birnn_wiki100k_en_zh.tar.gz"


ddir=datasets/
trainedmodelsdir=trained_models/

for u in $DATA_URLS
do
   wget "${u}" -P ${ddir}
   bn="$(basename $u)"
   tarfile="${ddir}${bn}" 
   #echo $tarfile 
   tar -xf $tarfile -C ${ddir}
   rm $tarfile 	 
done

mkdir ${trainedmodelsdir}

for u in $MODELS_URL
do
   wget "${u}" -P ${trainedmodelsdir}
done
