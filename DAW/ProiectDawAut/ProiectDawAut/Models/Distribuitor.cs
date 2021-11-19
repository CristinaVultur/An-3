using System;
using Microsoft.SqlServer.Server;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Web.Mvc;



namespace ProiectDawAut.Models
{
    public class Distribuitor
    {
        [Key]
        public int DistribuitorId { get; set; }
        [MinLength(2, ErrorMessage = "Nume prea scurt!"),
           MaxLength(200, ErrorMessage = "Nume prea lung!")]
        public string Nume { get; set; }       


        // many-to-one relationship
        public virtual ICollection<Bijuterii> Bijuterii { get; set; }
        // one-to one-relationship
        [Required]
        public virtual ContactInfo ContactInfo { get; set; }
    }
}