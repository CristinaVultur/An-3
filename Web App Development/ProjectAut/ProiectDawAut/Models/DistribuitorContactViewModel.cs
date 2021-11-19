using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace ProiectDawAut.Models
{
    public class DistribuitorContactViewModel
    {
        [MinLength(2, ErrorMessage = "Nume prea scurt!"),
           MaxLength(200, ErrorMessage = "Nume prea lung!")]
        public string Nume { get; set; }
        [RegularExpression(@"^07(\d{8})$", ErrorMessage = "Nr telefon invalid!")]
        public string PhoneNumber { get; set; }
        [Required, RegularExpression(@"^[1-9](\d{3})$", ErrorMessage = "An invalid!")]
        public int StartYear { get; set; }

        [Required, RegularExpression(@"^(0[1-9])|(1[012])$", ErrorMessage = "Luna invalida!")]
        public string StartMonth { get; set; }

        [Required, RegularExpression(@"^((0[1-9])|([12]\d)|(3[01]))$", ErrorMessage = "Data invalida!")]
        public string StartDay { get; set; }
        [MinLength(10, ErrorMessage = " prea scurt!")]
        public string Adresa { get; set; }
    }
}