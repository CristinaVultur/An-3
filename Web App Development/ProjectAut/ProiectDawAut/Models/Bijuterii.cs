using Microsoft.SqlServer.Server;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Data.Entity;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using ProiectDawAut.Models.MyValidation;

namespace ProiectDawAut.Models
{
    public class Bijuterii
    {
        [Key, Column("Id_Bijuterie")]
        public int IdBijuterie { get; set; }
        [MinLength(2, ErrorMessage = "Nume prea scurt!"),
            MaxLength(200, ErrorMessage = "Nume prea lung!")]
        public string Nume { get; set; }
        [MinLength(2, ErrorMessage = " prea scurt!"),
            MaxLength(200, ErrorMessage = " prea lung!")]
        public string Tip { get; set; }
        [TenMultipleValidator]
        public int Pret { get; set; }
        [MinLength(10, ErrorMessage = " prea scurt!")]
        public string Image { get; set; }

        // many to many
        public virtual ICollection<Comenzi> Comenzi { get; set; }
        // one to many
        [ForeignKey("Distribuitor")]
        public int DistribuitorId { get; set; }
        public virtual Distribuitor Distribuitor { get; set; }
        [NotMapped]
        public IEnumerable<SelectListItem> DistribuitorsList { get; set; }
    }

}
