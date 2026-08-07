#!/usr/bin/perl
use strict;
use warnings;

# Open and read the second table, storing sequences and their corresponding values in a hash
my %values;
open my $table2, '<', 'ref-SH2.txt' or die "Cannot open table2.txt: $!";
while (my $line = <$table2>) {
    chomp $line;
    my ($sequence, $value) = split /\t/, $line;
    $values{$sequence} = $value;
}
close $table2;

# Open the first table and output file
open my $table1, '<', 'stat.txt' or die "Cannot open table1.txt: $!";
open my $output, '>', 'stat.ref.txt' or die "Cannot open output.txt: $!";

# Read each line of the first table, find the corresponding value for the sequence, and add it as a new seventh column
while (my $line = <$table1>) {
    chomp $line;
    my @fields = split /\t/, $line;
    my $sequence = $fields[0];
    my $value = $values{$sequence} // '';
    print $output join("\t", @fields, $value), "\n";
}

close $table1;
close $output;

print "Process completed. Output saved to output.txt\n";
