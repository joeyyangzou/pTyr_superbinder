use strict;
use warnings;


# Open the file
open my $fh, '<', 'stat.ref.denoise.classification1.txt' or die "Could not open file: $!";

# Initialize arrays to store rows with Flag 1 and Flag 0
my @flag1_rows;
my @flag0_rows;

<$fh>;
# Read the file line by line
while (my $line = <$fh>) {
    chomp $line;
    my @cols = split(/\t/, $line);

    # Check Flag value and store in respective arrays
    if ($cols[4] == 1) {
        push @flag1_rows, $line;
    } elsif ($cols[4] == 0) {
        push @flag0_rows, $line;
    }
}

# Sort Flag 0 rows based on Diff_R4_R2
my @sorted_flag0 = sort { (split(/\t/, $a))[3] <=> (split(/\t/, $b))[3] } @flag0_rows;

my $positive_count = scalar @flag1_rows;
@sorted_flag0 = @sorted_flag0[0..$positive_count-1];

# Write to positive.txt
open(my $pos_fh, '>', 'positive') or die "Could not open file 'positive.txt' $!";
print $pos_fh "sequence\tlabel\n";
foreach my $row (@flag1_rows) {
    my @p=split /\t/,$row;
    print $pos_fh "$p[0]\t$p[4]\n";
}
close $pos_fh;

# Write to negative.txt
open(my $neg_fh, '>', 'negative') or die "Could not open file 'negative.txt' $!";
print $neg_fh "sequence\tlabel\n";
foreach my $row (@sorted_flag0) {
    my @q=split /\t/,$row;
    print $neg_fh "$q[0]\t$q[4]\n";
}
close $neg_fh;

# Close the input file
close $fh;
