#!/usr/bin/env perl
use strict;
use warnings;

# Each input file contains one variable-region peptide per read. When no files
# are supplied, all *.short files in the working directory are used.
my @files = @ARGV ? @ARGV : glob "*.short";
die "No input files supplied and no *.short files found\n" unless @files;

my %sequences;
for my $file_index (0 .. $#files) {
    open my $input_handle, '<', $files[$file_index]
        or die "Cannot open $files[$file_index]: $!\n";
    while (my $line = <$input_handle>) {
        chomp $line;
        $line =~ s/\r$//;
        next if $line eq '';
        $sequences{$line}[$file_index]++;
    }
    close $input_handle;
}

for my $sequence (sort keys %sequences) {
    print $sequence;
    my $total = 0;
    for my $file_index (0 .. $#files) {
        my $count = $sequences{$sequence}[$file_index] // 0;
        print "\t$count";
        $total += $count;
    }
    my $sequence_length = length $sequence;
    $sequence_length-- if index($sequence, '-') >= 0;
    print "\t$sequence_length\t$total\n";
}
