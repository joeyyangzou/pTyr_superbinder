#!/usr/bin/env perl
use strict;
use warnings;

my ($input_file, $output_file) = @ARGV;
die "Usage: $0 input.short output.stat.tsv\n" unless defined $output_file;

open my $input_handle, '<', $input_file
    or die "Cannot open $input_file: $!\n";

my %counts;
while (my $line = <$input_handle>) {
    chomp $line;
    $line =~ s/\r$//;
    next if $line eq '';
    $counts{$line}++;
}
close $input_handle;

open my $output_handle, '>', $output_file
    or die "Cannot write $output_file: $!\n";
for my $sequence (sort keys %counts) {
    print {$output_handle} "$sequence\t$counts{$sequence}\n";
}
close $output_handle;
