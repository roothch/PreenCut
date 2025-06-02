#!/usr/bin/perl -w

# Author: Ulf Hermjakob
# Release date: October 13, 2019

# Usage: string-distance.pl {-lc1 <language-code>} {-lc2 <language-code>} < STDIN > STDOUT
# Example: string-distance.pl -lc1 rus -lc2 ukr < STDIN > STDOUT
# Example: string-distance.pl < ../test/string-similarity-test-input.txt
# Input format: two strings per line (tab-separated, in Latin script)
#    Strings in non-Latin scripts should first be romanized. (Recommended script: uroman.pl)
# Output format: repetition of the two input strings, plus the string distance between them (tab-separated).
#    Additional output meta info lines at the top are marked with an initial #.
#
# The script uses data from a string-distance-cost-rules file that lists costs,
# where the default cost is "1" with lower costs for differences in vowels,
# duplicate consonants, "f" vs. "ph" etc.
# Language cost rules can be language-specific and context-sensitive.

$|=1;

use FindBin;
use Cwd "abs_path";
use File::Basename qw(dirname);
use File::Spec;

my $bin_dir = abs_path(dirname($0));
my $root_dir = File::Spec->catfile($bin_dir, File::Spec->updir());
my $data_dir = File::Spec->catfile($root_dir, "data");
my $lib_dir = File::Spec->catfile($root_dir, "lib");

use lib "$FindBin::Bin/../lib";
use List::Util qw(min max);
use NLP::utilities;
use NLP::stringDistance;
$util = NLP::utilities;
$sd = NLP::stringDistance;
$verbose = 0;
$separator = "\t";

$cost_rule_filename = File::Spec->catfile($data_dir, "string-distance-cost-rules.txt");

$lang_code1 = "eng";
$lang_code2 = "eng";
%ht = ();

while (@ARGV) {
   $arg = shift @ARGV;
   if ($arg =~ /^-+lc1$/) {
      $lang_code_candidate = shift @ARGV;
      $lang_code1 = $lang_code_candidate if $lang_code_candidate =~ /^[a-z]{3,3}$/;
   } elsif ($arg =~ /^-+lc2$/) {
      $lang_code_candidate = shift @ARGV;
      $lang_code2 = $lang_code_candidate if $lang_code_candidate =~ /^[a-z]{3,3}$/;
   } elsif ($arg =~ /^-+(v|verbose)$/) {
      $verbose = shift @ARGV;
   } else {
      print STDERR "Ignoring unrecognized arg $arg\n";
   }
}

$sd->load_string_distance_data($cost_rule_filename, *ht, $verbose);
print STDERR "Loaded resources.\n" if $verbose;

my $chart_id = 0;
my $line_number = 0;
print "# Lang-code-1: $lang_code1 Lang-code-2: $lang_code2\n";
while (<>) {
   $line_number++;
   if ($verbose) {
      if ($line_number =~ /000$/) {
         if ($line_number =~ /0000$/) {
	    print STDERR $line_number;
         } else {
	    print STDERR ".";
         }
      }
   }
   my $line = $_;
   $line =~ s/^\xEF\xBB\xBF//;
   next if $line =~ /^\s*(\#.*)?$/;
   my $s1;
   my $s2;
   if (($s1, $s2) = ($line =~ /^("(?:\\"|[^"])*"|\S+)$separator("(?:\\"|[^"])*"|\S+)\s*$/)) {
      $s1 = $util->dequote_string($s1);
      $s2 = $util->dequote_string($s2);
   } elsif ($line =~ /^\s*(#.*)$/) {
   } else {
      print STDERR "Could not process line $line_number: $line" if $verbose;
      print "\n";
      next;
   }

   $cost = $sd->quick_romanized_string_distance_by_chart($s1, $s2, *ht, "", $lang_code1, $lang_code2);
   print "$s1\t$s2\t$cost\n";
}
print STDERR "\n" if $verbose;

exit 0;

