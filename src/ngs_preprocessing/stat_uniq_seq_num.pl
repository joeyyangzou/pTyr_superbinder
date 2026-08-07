#!/usr/bin/perl
use strict;
use warnings;


my %sequences;
#my @files = ('H9-K9me3_R1.final.fa.pep.filter.short', 'Cbx1-H9.final.fa.pep.filter.fa', 'file3', 'file4', 'file5', 'file6');
my @files = glob "*.haoqiang";

for my $i (0..$#files) {
    open my $fh, '<', $files[$i] or die $!;
    while (<$fh>) {
        chomp;
        $sequences{$_}[$i]++;
    }
    close $fh;
}

for my $seq (keys %sequences) {
    print $seq;
    for my $i (0..$#files) {
        print "\t", ($sequences{$seq}[$i] // 0);
    }
    my @counts = map { $_ // 0 } @{$sequences{$seq}};
    my $length = length $seq;
    my $sum = 0;
    $sum += $_ for @counts;

    print "\t", length($seq)-1, "\t",$sum,"\n";
}



