using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace ProiectDaw.Models
{
    public class ContactInfo
    {
        [Key]
        public int ContactInfoId { get; set; }
        public string PhoneNumber { get; set; }
        // one-to one-relationship
        public virtual User User { get; set; }
    }
}