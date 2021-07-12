#! /bin/sh

#urls for data download
DATA_URLS="https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_mlqe.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_mlqe.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_mlqe.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_mlqe.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_mlqe.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/et_en_25k_wiki.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ro_en_100k_wiki.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/si_en_100k_wiki.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/ne_en_100k_wiki.tar.gz 
https://www.quest.dcs.shef.ac.uk/dq_student_birnn/en_zh_100k_wiki.tar.gz"


ddir=datasets/

for u in $DATA_URLS
do
   wget "${u}" -P ${ddir}
   bn="$(basename $u)"
   tarfile="${ddir}${bn}" 
   #echo $tarfile 
   tar -xf $tarfile -C ${ddir}
   rm $tarfile 	 
done

