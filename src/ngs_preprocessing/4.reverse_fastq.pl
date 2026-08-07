use strict;
use warnings;

my $infile = $ARGV[0];#reverse.fq
my $outfile = $ARGV[1];#reverse.re.fq

open IN,  "<", $infile;
open OUT, ">", $outfile;

while (my $line = <IN>){
	if($line =~ /^@/){
		my $n_line = <IN>;
		chomp $n_line;
		my $revcom=reverse $n_line;
		$revcom =~ tr/ACGTacgt/TGCAtgca/;
		my $nn_line = <IN>;
		my $nnn_line = <IN>;
		chomp $nnn_line;
		my $revcomm=reverse $nnn_line;
		print OUT $line,$revcom,"\n",$nn_line,$revcomm,"\n";
	}
}

