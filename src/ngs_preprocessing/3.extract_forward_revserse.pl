use strict;
use warnings;

my $infile = $ARGV[0];#rmadaptor.fq
my $outfile_forward = $ARGV[1];#forward.fq
my $outfile_reverse = $ARGV[2];#reverse.fq

open IN,  "<", $infile;
open OUT1, ">", $outfile_forward;
open OUT2, ">", $outfile_reverse;

while (my $line = <IN>){
	   if($line =~ /^@/){
		         my $n_line = <IN>;
			 my $nn_line = <IN>;
		         my $nnn_line = <IN>;
		if ($n_line=~/^CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA$/){
		        print OUT1 $line,$n_line,$nn_line,$nnn_line;
		}else{
			print OUT2 $line,$n_line,$nn_line,$nnn_line;
		}
	}
}
close IN;
close OUT1;
close OUT2;
