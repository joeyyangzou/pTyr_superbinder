# ANCHOR: AI-NGS Consensus Heuristic for Optimized Refinement of High-Affinity Binders from Affinity Maturation
The development of high-affinity binding variants for post-translational modification driven by artificial intelligence

1. Use the Flash software to perform double-end sequence splicing with default parameters. Command: flash -o test -t 8 SH2_R1.fastq SH2_R2.fastq (pYS2.extendedFrags.fastq)
2. Based on the spliced sequences, match the forward and reverse sequences of CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA and TGTTTCAAACTGGGC.*GTACCACTCTTCTGCCTG. Delete the sequences before and after the specified patterns, and retain the patterns and the parts between them (perl 3.rmadpator.pl pYS2.extendedFrags.fastq pYS2.extendedFrags.rmadaptor.fastq)
3. According to the fixed sequences before and after the sequencing template sequence, determine the forward and reverse directions of the sequencing data ^CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA$. Each sample is split into a forward sample file and a reverse sample file. Sequencing data that do not meet the sequencing template are deleted (here, "not meeting" refers to one or more base mismatches. This condition is a bit strict and can be appropriately relaxed. Sequencing data with two or fewer mismatches compared to the sequencing template can also be retained). 5.extract_forward_revserse.pl
4. Reverse complement the reverse samples to obtain the forward samples, and then merge them with the original forward samples to get the final DNA fq file. 6.reverse_fastq.pl  cat
5. Convert the fq file to a fa file, and filter out sequences that do not match the beginning and end of the sequencing template (next unless /^CAAAACAAGAA/; ) next unless /TGCCCTGAC$/;) And translate it into a protein FASTA file. (TAA>_; TAG>q; TGA>w; .N.>X; N.. >X; .. N>X) 8. fastq2fasta.sh 9. dna2pep.pl for i in *.final.fq; do sed -n '1~4s/^@/>/p; 2~4p' $i > ${i%.*}.fa; done
6. Filter the translated protein fa file based on the protein sequencing template with the condition "^QNKKKVEEVLEEEE.*EKVLDRRVVKGKVEYLLK.*PEENLDCP". for i in *final.fa.new.pep; do grep "^NKKKVEEVLEEEE.*EKVLDRRVVKGKVEYLLK.*PEENLDCP" $i > $i.filter; 
7. Remove the fixed sequence and connect the two variable sequences with a hyphen. for i in *.final.fa.pep.filter; do perl -pe 's/^NKKKVEEVLEEEE|-EKVLDRRVVKGKVEYLLK|PEENLDCP$//g; s/EKVLDRRVVKGKVEYLLK/-/g' $i > $i.short; done
8. for i in *.final.fa.pep.filter; do perl stat_uniq_pep_num.pl $i $i.stat.csv; done
9. Count the number of unique sequences in each sample, the length of the sequences, and the total sum of sequences in all samples. perl stat_uniq_seq_num.pl > stat.txt
10. perl join.pl stat.txt -> stat.ref.txt
11. python sequence_denoise.py > stat.ref.denoise.txt 
12. awk '$4 > 10 || $6 > 10' stat.ref.denoise.txt > stat.ref.denoise.10.txt
13. Run the script to calculate the average copy number after statistical denoising: python stat_average_copy.py
14. Run the script to compare R2 and R4: python calculate_classification.py
15. Run classification_positive_negative.pl first to check the number of positives, then modify the numbers inside to ensure the number of negatives is the same.
16. Add the header "sequence label" to both positive and negative files.
17. python calculate_regression.py
18. python split_pos_neg_regression.py
19. cat positive_samples.txt negative_samples.txt | sed '1d' > out.txt 
20. python split_train_test.py
21. `cut -f 1,4 stat.ref.denoise.10.regression.txt > out` Remember to change the header to `sequence value` 
22. conda activate tensorflow2.4 nohup python classification_Multi-thread_new.py output_T.txt output_T_predict.txt --batch_size 10240 &
23. cut -f 1 split_all2/predict_all.out > pass_classification99_seq
24. python regression_multi_thread.py pass_classification99_seq pass_classification99_seq_regression_score
