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
    public class Comenzi
    {
        [Key, Column("Id_Comanda")]
        public int IdComanda { get; set; }
        [Range(0, int.MaxValue, ErrorMessage = "Comanda nu poate fi zero lei!")]
        public int Total { get; set; }
        //many to many
        public virtual ICollection<Bijuterii> Bijuterii { get; set; }
        // one to many
       
        public string UserId { get; set; }
        public virtual ApplicationUser User { get; set; }
        //public virtual UserViewModel User { get; set; }
        [NotMapped]
        public IEnumerable<CheckBoxViewModel> BijuteriiList { get; set; }
        [NotMapped]
        public IEnumerable<SelectListItem> UserList { get; set; }

    }
}