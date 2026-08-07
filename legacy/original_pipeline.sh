flash -o test -t 8 SH2_R1.fastq SH2_R2.fastq
for i in *.extendedFrags.rmadaptor.reverse.re.fastq;do cat ${i%%.*}.extendedFrags.rmadaptor.forward.fastq $i > ${i%%.*}.final.fq;done

for i in *.final.fq; do sed -n '1~4s/^@/>/p;2~4p' $i > ${i%.*}.fa;done
for i in *final.fa.pep;do grep "^NKKKVEEVLEEEE.*EKVLDRRVVKGKVEYLLK.*PEENLDCP" $i > $i.filter;done

for i in *.final.fa.pep.filter;do perl -pe 's/^NKKKVEEVLEEEE|-EKVLDRRVVKGKVEYLLK|PEENLDCP$//g; s/EKVLDRRVVKGKVEYLLK/-/g' $i > $i.short;done

for i in *.final.fa.pep.filter;do perl stat_uniq_pep_num.pl $i $i.stat.csv;done
