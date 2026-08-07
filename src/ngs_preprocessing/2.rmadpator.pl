#!/usr/bin/perl
use strict;
use warnings;

# Input and output filenames
my ($input_file,$output_file) = @ARGV;

# Open input and output files
open(my $in, '<', $input_file) or die "Cannot open file $input_file: $!";
open(my $out, '>', $output_file) or die "Cannot open file $output_file: $!";

# Read and process each record (every four lines is one record)
while (my $header = <$in>) {
    my $sequence = <$in>;
    my $plus = <$in>;
    my $quality = <$in>;
chomp $header;
chomp $sequence;
chomp $plus;
chomp $quality;
    # Delete the sequence before and after the specified pattern, keeping the pattern and everything in between
    if ($sequence =~ /(CAGGCAGAAGAGTGGTAC.*GCCCAGTTTGAAACA)/) {
        my $match = $1;
        my $start = $-[0];
        my $end = $+[0];
        $sequence = $match;
        $quality = substr($quality, $start, $end - $start);
    } elsif ($sequence =~ /(TGTTTCAAACTGGGC.*GTACCACTCTTCTGCCTG)/) {
        my $match = $1;
        my $start = $-[0];
        my $end = $+[0];
        $sequence = $match;
        $quality = substr($quality, $start, $end - $start);
    }

    # Output the processed record
    print $out $header . "\n";
    print $out $sequence . "\n";
    print $out $plus . "\n";
    print $out $quality . "\n";
}

# Close file handles
close($in);
close($out);

print "Processing complete. Output saved to $output_file\n";
