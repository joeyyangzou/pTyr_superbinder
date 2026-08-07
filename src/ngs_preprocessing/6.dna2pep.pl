#!/usr/bin/perl

use strict;
use warnings;


open (FA,"$ARGV[0]");
open (FB,">$ARGV[1]");


while (<FA>){
        chomp;
        #print FB $_ "\n" if /^>/;
        next if /^>/;
        my $protein;
        for(my $i=0; $i < (length($_) - 4) ; $i += 3) {
                $protein .= codonaa( substr($_,$i,3) );
        }
        print FB $protein,"\n";
}


sub codonaa { 
    my($codon) = @_; 
       if ( $codon =~ /TCA/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /TCC/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /TCG/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /TCT/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /TCN/i )    { return 'S' }    # Serine 
	elsif ( $codon =~ /TTC/i )    { return 'F' }    # Phenylalanine 
    elsif ( $codon =~ /TTT/i )    { return 'F' }    # Phenylalanine 
    elsif ( $codon =~ /TTA/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /TTG/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /TTN/i )    { return 'X' }    # Leucine 
	elsif ( $codon =~ /TAC/i )    { return 'Y' }    # Tyrosine 
    elsif ( $codon =~ /TAT/i )    { return 'Y' }    # Tyrosine 
    elsif ( $codon =~ /TAN/i )    { return 'X' }    # Tyrosine 
	elsif ( $codon =~ /TAA/i )    { return '_' }    # Stop 
    elsif ( $codon =~ /TAG/i )    { return 'q' }    # Stop Q
    elsif ( $codon =~ /TGC/i )    { return 'C' }    # Cysteine 
    elsif ( $codon =~ /TGT/i )    { return 'C' }    # Cysteine 
    elsif ( $codon =~ /TGN/i )    { return 'X' }    # Cysteine 
    elsif ( $codon =~ /TGA/i )    { return 'w' }    # Stop  W
    elsif ( $codon =~ /TGG/i )    { return 'W' }    # Tryptophan 
    elsif ( $codon =~ /CTA/i )    { return 'L' }    # Leucine 
	elsif ( $codon =~ /CTC/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /CTG/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /CTT/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /CTN/i )    { return 'L' }    # Leucine 
    elsif ( $codon =~ /CCA/i )    { return 'P' }    # Proline 
    elsif ( $codon =~ /CCC/i )    { return 'P' }    # Proline 
    elsif ( $codon =~ /CCG/i )    { return 'P' }    # Proline 
    elsif ( $codon =~ /CCT/i )    { return 'P' }    # Proline 
    elsif ( $codon =~ /CCN/i )    { return 'P' }    # Proline 
    elsif ( $codon =~ /CAC/i )    { return 'H' }    # Histidine 
    elsif ( $codon =~ /CAT/i )    { return 'H' }    # Histidine 
    elsif ( $codon =~ /CAA/i )    { return 'Q' }    # Glutamine 
    elsif ( $codon =~ /CAG/i )    { return 'Q' }    # Glutamine 
    elsif ( $codon =~ /CAN/i )    { return 'X' }    # Glutamine 
	elsif ( $codon =~ /CGA/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /CGC/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /CGG/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /CGT/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /CGN/i )    { return 'R' }    # Arginine 
	elsif ( $codon =~ /ATA/i )    { return 'I' }    # Isoleucine 
    elsif ( $codon =~ /ATC/i )    { return 'I' }    # Isoleucine 
	elsif ( $codon =~ /ATT/i )    { return 'I' }    # Isoleucine 
    elsif ( $codon =~ /ATN/i )    { return 'X' }    # Isoleucine 
	elsif ( $codon =~ /ATG/i )    { return 'M' }    # Methionine 
    elsif ( $codon =~ /ACA/i )    { return 'T' }    # Threonine 
    elsif ( $codon =~ /ACC/i )    { return 'T' }    # Threonine 
    elsif ( $codon =~ /ACG/i )    { return 'T' }    # Threonine 
    elsif ( $codon =~ /ACT/i )    { return 'T' }    # Threonine 
    elsif ( $codon =~ /ACN/i )    { return 'T' }    # Threonine 
	elsif ( $codon =~ /AAC/i )    { return 'N' }    # Asparagine 
    elsif ( $codon =~ /AAT/i )    { return 'N' }    # Asparagine 
	elsif ( $codon =~ /AAA/i )    { return 'K' }    # Lysine 
    elsif ( $codon =~ /AAG/i )    { return 'K' }    # Lysine 
	elsif ( $codon =~ /AAN/i )    { return 'X' } 
    elsif ( $codon =~ /AGC/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /AGT/i )    { return 'S' }    # Serine 
    elsif ( $codon =~ /AGA/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /AGG/i )    { return 'R' }    # Arginine 
    elsif ( $codon =~ /AGN/i )    { return 'X' }    # Arginine 
    elsif ( $codon =~ /GTA/i )    { return 'V' }    # Valine 
    elsif ( $codon =~ /GTC/i )    { return 'V' }    # Valine 
    elsif ( $codon =~ /GTG/i )    { return 'V' }    # Valine 
    elsif ( $codon =~ /GTT/i )    { return 'V' }    # Valine 
    elsif ( $codon =~ /GTN/i )    { return 'V' }    # Valine 
    elsif ( $codon =~ /GCA/i )    { return 'A' }    # Alanine 
    elsif ( $codon =~ /GCC/i )    { return 'A' }    # Alanine 
    elsif ( $codon =~ /GCG/i )    { return 'A' }    # Alanine 
    elsif ( $codon =~ /GCT/i )    { return 'A' }    # Alanine
	elsif ( $codon =~ /GCN/i )    { return 'A' }    # Alanine
    elsif ( $codon =~ /GAC/i )    { return 'D' }    # Aspartic Acid 
    elsif ( $codon =~ /GAT/i )    { return 'D' }    # Aspartic Acid 
    elsif ( $codon =~ /GAA/i )    { return 'E' }    # Glutamic Acid 
	elsif ( $codon =~ /GAG/i )    { return 'E' }    # Glutamic Acid 
  	elsif ( $codon =~ /GAN/i )    { return 'X' }
    elsif ( $codon =~ /GGA/i )    { return 'G' }    # Glycine 
    elsif ( $codon =~ /GGC/i )    { return 'G' }    # Glycine 
    elsif ( $codon =~ /GGG/i )    { return 'G' }    # Glycine 
	elsif ( $codon =~ /GGT/i )    { return 'G' }    # Glycine 
    elsif ( $codon =~ /GGN/i )    { return 'G' }    # Glycine 
    elsif ( $codon =~ /.N./i )    { return 'X' }     
    elsif ( $codon =~ /N../i)     { return 'X' }    
	elsif ( $codon =~ /NN./i )    { return 'X' }    
	elsif ( $codon =~ /..N/i )    { return 'X' }      
	elsif ( $codon =~ /.NN/i )    { return 'X' }      
	elsif ( $codon =~ /N.N/i)     { return 'X' }
	elsif ( $codon =~ /NNN/i)     { return 'X' }
	else  { 
		    print STDERR "Bad codon \"$codon\"!!\n"; 
            exit; 
    }
}
