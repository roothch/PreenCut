#!/usr/bin/perl -w

# uroman  Nov. 12, 2015 - July 25, 2016
# version v0.7
# Author: Ulf Hermjakob

# Usage: uroman-quick.pl {-l [tur|uig|ukr|yid]} < STDIN
# currently only for Arabic script languages, incl. Uyghur

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
use NLP::Romanizer;
use NLP::UTF8;
$romanizer = NLP::Romanizer;
%ht = ();
$lang_code = "";

while (@ARGV) {
   $arg = shift @ARGV;
   if ($arg =~ /^-+(l|lc|lang-code)$/) {
      $lang_code = lc (shift @ARGV || "")
   } else {
      print STDERR "Ignoring unrecognized arg $arg\n";
   }
}

$romanization_table_arabic_block_filename = File::Spec->catfile($data_dir, "romanization-table-arabic-block.txt");
$romanization_table_filename = File::Spec->catfile($data_dir, "romanization-table.txt");

$romanizer->load_romanization_table(*ht, $romanization_table_arabic_block_filename);
$romanizer->load_romanization_table(*ht, $romanization_table_filename);

$line_number = 0;
while (<>) {
   $line_number++;
   my $line = $_;
   print $romanizer->quick_romanize($line, $lang_code, *ht) . "\n";
   if ($line_number =~ /0000$/) {
      print STDERR $line_number;
   } elsif ($line_number =~ /000$/) {
      print STDERR ".";
   }
}
print STDERR "\n";

exit 0;

