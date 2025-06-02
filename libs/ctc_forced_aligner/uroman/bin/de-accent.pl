#!/usr/bin/perl -w

sub print_version {
   print STDERR "$0 version 1.1\n";
   print STDERR "   Author: Ulf Hermjakob\n";
   print STDERR "   Last changed: March 14, 2011\n";
}

sub print_usage {
   print STDERR "$0 [options] < with_accents.txt > without_accents.txt\n";
   print STDERR "   -h or -help\n";
   print STDERR "   -v or -version\n";
}

sub de_accent_string {
   local($s) = @_;

   # $s =~ tr/A-Z/a-z/;
   unless (0) {
      # Latin-1
      if ($s =~ /\xC3[\x80-\xBF]/) {
         $s =~ s/(À|Á|Â|Ã|Ä|Å)/A/g;
         $s =~ s/Æ/Ae/g;
         $s =~ s/Ç/C/g;
         $s =~ s/Ð/D/g;
         $s =~ s/(È|É|Ê|Ë)/E/g;
         $s =~ s/(Ì|Í|Î|Ï)/I/g;
         $s =~ s/Ñ/N/g;
         $s =~ s/(Ò|Ó|Ô|Õ|Ö|Ø)/O/g;
         $s =~ s/(Ù|Ú|Û|Ü)/U/g;
         $s =~ s/Þ/Th/g;
         $s =~ s/Ý/Y/g;
         $s =~ s/(à|á|â|ã|ä|å)/a/g;
         $s =~ s/æ/ae/g;
         $s =~ s/ç/c/g;
         $s =~ s/(è|é|ê|ë)/e/g;
         $s =~ s/(ì|í|î|ï)/i/g;
         $s =~ s/ð/d/g;
         $s =~ s/ñ/n/g;
         $s =~ s/(ò|ó|ô|õ|ö)/o/g;
         $s =~ s/ß/ss/g;
         $s =~ s/þ/th/g;
         $s =~ s/(ù|ú|û|ü)/u/g;
         $s =~ s/(ý|ÿ)/y/g;
      }
      # Latin Extended-A
      if ($s =~ /[\xC4-\xC5][\x80-\xBF]/) {
         $s =~ s/(Ā|Ă|Ą)/A/g;
         $s =~ s/(ā|ă|ą)/a/g;
         $s =~ s/(Ć|Ĉ|Ċ|Č)/C/g;
         $s =~ s/(ć|ĉ|ċ|č)/c/g;
         $s =~ s/(Ď|Đ)/D/g;
         $s =~ s/(ď|đ)/d/g;
         $s =~ s/(Ē|Ĕ|Ė|Ę|Ě)/E/g;
         $s =~ s/(ē|ĕ|ė|ę|ě)/e/g;
         $s =~ s/(Ĝ|Ğ|Ġ|Ģ)/G/g;
         $s =~ s/(ĝ|ğ|ġ|ģ)/g/g;
         $s =~ s/(Ĥ|Ħ)/H/g;
         $s =~ s/(ĥ|ħ)/h/g;
         $s =~ s/(Ĩ|Ī|Ĭ|Į|İ)/I/g;
         $s =~ s/(ĩ|ī|ĭ|į|ı)/i/g;
         $s =~ s/Ĳ/Ij/g;
         $s =~ s/ĳ/ij/g;
         $s =~ s/Ĵ/J/g;
         $s =~ s/ĵ/j/g;
         $s =~ s/Ķ/K/g;
         $s =~ s/(ķ|ĸ)/k/g;
         $s =~ s/(Ĺ|Ļ|Ľ|Ŀ|Ł)/L/g;
         $s =~ s/(ļ|ľ|ŀ|ł)/l/g;
         $s =~ s/(Ń|Ņ|Ň|Ŋ)/N/g;
         $s =~ s/(ń|ņ|ň|ŉ|ŋ)/n/g;
         $s =~ s/(Ō|Ŏ|Ő)/O/g;
         $s =~ s/(ō|ŏ|ő)/o/g;
         $s =~ s/Œ/Oe/g;
         $s =~ s/œ/oe/g;
         $s =~ s/(Ŕ|Ŗ|Ř)/R/g;
         $s =~ s/(ŕ|ŗ|ř)/r/g;
         $s =~ s/(Ś|Ŝ|Ş|Š)/S/g;
         $s =~ s/(ś|ŝ|ş|š|ſ)/s/g;
         $s =~ s/(Ţ|Ť|Ŧ)/T/g;
         $s =~ s/(ţ|ť|ŧ)/t/g;
         $s =~ s/(Ũ|Ū|Ŭ|Ů|Ű|Ų)/U/g;
         $s =~ s/(ũ|ū|ŭ|ů|ű|ų)/u/g;
         $s =~ s/Ŵ/W/g;
         $s =~ s/ŵ/w/g;
         $s =~ s/(Ŷ|Ÿ)/Y/g;
         $s =~ s/ŷ/y/g;
         $s =~ s/(Ź|Ż|Ž)/Z/g;
         $s =~ s/(ź|ż|ž)/z/g;
      }
      # Latin Extended Additional
      if ($s =~ /\xE1[\xB8-\xBF][\x80-\xBF]/) {
          $s =~ s/(ḁ|ạ|ả|ấ|ầ|ẩ|ẫ|ậ|ắ|ằ|ẳ|ẵ|ặ|ẚ)/a/g;
          $s =~ s/(ḃ|ḅ|ḇ)/b/g;
          $s =~ s/(ḉ)/c/g;
          $s =~ s/(ḋ|ḍ|ḏ|ḑ|ḓ)/d/g;
          $s =~ s/(ḕ|ḗ|ḙ|ḛ|ḝ|ẹ|ẻ|ẽ|ế|ề|ể|ễ|ệ)/e/g;
          $s =~ s/(ḟ)/f/g;
          $s =~ s/(ḡ)/g/g;
          $s =~ s/(ḣ|ḥ|ḧ|ḩ|ḫ)/h/g;
          $s =~ s/(ḭ|ḯ|ỉ|ị)/i/g;
          $s =~ s/(ḱ|ḳ|ḵ)/k/g;
          $s =~ s/(ḷ|ḹ|ḻ|ḽ)/l/g;
          $s =~ s/(ḿ|ṁ|ṃ)/m/g;
          $s =~ s/(ṅ|ṇ|ṉ|ṋ)/m/g;
          $s =~ s/(ọ|ỏ|ố|ồ|ổ|ỗ|ộ|ớ|ờ|ở|ỡ|ợ|ṍ|ṏ|ṑ|ṓ)/o/g;
          $s =~ s/(ṕ|ṗ)/p/g;
          $s =~ s/(ṙ|ṛ|ṝ|ṟ)/r/g;
          $s =~ s/(ṡ|ṣ|ṥ|ṧ|ṩ|ẛ)/s/g;
          $s =~ s/(ṫ|ṭ|ṯ|ṱ)/t/g;
          $s =~ s/(ṳ|ṵ|ṷ|ṹ|ṻ|ụ|ủ|ứ|ừ|ử|ữ|ự)/u/g;
          $s =~ s/(ṽ|ṿ)/v/g;
          $s =~ s/(ẁ|ẃ|ẅ|ẇ|ẉ|ẘ)/w/g;
          $s =~ s/(ẋ|ẍ)/x/g;
          $s =~ s/(ẏ|ỳ|ỵ|ỷ|ỹ|ẙ)/y/g;
          $s =~ s/(ẑ|ẓ|ẕ)/z/g;
          $s =~ s/(Ḁ|Ạ|Ả|Ấ|Ầ|Ẩ|Ẫ|Ậ|Ắ|Ằ|Ẳ|Ẵ|Ặ)/A/g;
          $s =~ s/(Ḃ|Ḅ|Ḇ)/B/g;
          $s =~ s/(Ḉ)/C/g;
          $s =~ s/(Ḋ|Ḍ|Ḏ|Ḑ|Ḓ)/D/g;
          $s =~ s/(Ḕ|Ḗ|Ḙ|Ḛ|Ḝ|Ẹ|Ẻ|Ẽ|Ế|Ề|Ể|Ễ|Ệ)/E/g;
          $s =~ s/(Ḟ)/F/g;
          $s =~ s/(Ḡ)/G/g;
          $s =~ s/(Ḣ|Ḥ|Ḧ|Ḩ|Ḫ)/H/g;
          $s =~ s/(Ḭ|Ḯ|Ỉ|Ị)/I/g;
          $s =~ s/(Ḱ|Ḳ|Ḵ)/K/g;
          $s =~ s/(Ḷ|Ḹ|Ḻ|Ḽ)/L/g;
          $s =~ s/(Ḿ|Ṁ|Ṃ)/M/g;
          $s =~ s/(Ṅ|Ṇ|Ṉ|Ṋ)/N/g;
          $s =~ s/(Ṍ|Ṏ|Ṑ|Ṓ|Ọ|Ỏ|Ố|Ồ|Ổ|Ỗ|Ộ|Ớ|Ờ|Ở|Ỡ|Ợ)/O/g;
          $s =~ s/(Ṕ|Ṗ)/P/g;
          $s =~ s/(Ṙ|Ṛ|Ṝ|Ṟ)/R/g;
          $s =~ s/(Ṡ|Ṣ|Ṥ|Ṧ|Ṩ)/S/g;
          $s =~ s/(Ṫ|Ṭ|Ṯ|Ṱ)/T/g;
          $s =~ s/(Ṳ|Ṵ|Ṷ|Ṹ|Ṻ|Ụ|Ủ|Ứ|Ừ|Ử|Ữ|Ự)/U/g;
          $s =~ s/(Ṽ|Ṿ)/V/g;
          $s =~ s/(Ẁ|Ẃ|Ẅ|Ẇ|Ẉ)/W/g;
          $s =~ s/(Ẍ)/X/g;
          $s =~ s/(Ẏ|Ỳ|Ỵ|Ỷ|Ỹ)/Y/g; 
          $s =~ s/(Ẑ|Ẓ|Ẕ)/Z/g;
      }
      # Greek letters
      if ($s =~ /\xCE[\x86-\xAB]/) {
          $s =~ s/ά/α/g;
          $s =~ s/έ/ε/g;
          $s =~ s/ί/ι/g;
          $s =~ s/ϊ/ι/g;
          $s =~ s/ΐ/ι/g;
          $s =~ s/ό/ο/g;
          $s =~ s/ύ/υ/g;
          $s =~ s/ϋ/υ/g;
          $s =~ s/ΰ/υ/g;
          $s =~ s/ώ/ω/g;
          $s =~ s/Ά/Α/g;
          $s =~ s/Έ/Ε/g;
          $s =~ s/Ή/Η/g;
          $s =~ s/Ί/Ι/g;
          $s =~ s/Ϊ/Ι/g;
          $s =~ s/Ύ/Υ/g;
          $s =~ s/Ϋ/Υ/g;
          $s =~ s/Ώ/Ω/g;
      }
      # Cyrillic letters
      if ($s =~ /\xD0[\x80-\xAF]/) {
          $s =~ s/Ѐ/Е/g;
          $s =~ s/Ё/Е/g;
          $s =~ s/Ѓ/Г/g;
          $s =~ s/Ќ/К/g;
          $s =~ s/Ѝ/И/g;
          $s =~ s/Й/И/g;
          $s =~ s/ѐ/е/g;
          $s =~ s/ё/е/g;
          $s =~ s/ѓ/г/g;
          $s =~ s/ќ/к/g;
          $s =~ s/ѝ/и/g;
          $s =~ s/й/и/g;
      }
   }
   return $s;
}

while (@ARGV) {
   $arg = shift @ARGV;
   if ($arg =~ /^-*(h|help)$/i) {
      &print_usage;
      exit 1;
   } elsif ($arg =~ /^-*(v|version)$/i) {
      &print_version;
      exit 1;
   } else {
      print STDERR "Ignoring unrecognized argument $arg\n";
   }
}

$line_number = 0;
while (<>) {
   $line_number++;
   print &de_accent_string($_);
}
exit 0;

