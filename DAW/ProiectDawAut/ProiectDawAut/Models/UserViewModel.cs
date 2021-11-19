
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace ProiectDawAut.Models
{
    public class UserViewModel
    {
        public ApplicationUser User { get; set; }
        public string RoleName { get; set; }

        //public virtual ICollection<Comenzi> Comenzi { get; set; }
        // one-to one-relationship
        //[Required]
        //public virtual ContactInfo ContactInfo { get; set; }
    }
}