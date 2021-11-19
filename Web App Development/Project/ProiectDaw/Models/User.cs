using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace ProiectDaw.Models
{
    public class User
    {
        [Key]
        public int UserId { get; set; }
        public string Username { get; set; }

        public string Password { get; set; }
        // many-to-one relationship
        public virtual ICollection<Comenzi> Comenzi { get; set; }
        // one-to one-relationship
        [Required]
        public virtual ContactInfo ContactInfo { get; set; }
    }
}