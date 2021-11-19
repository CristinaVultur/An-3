using System;
using Microsoft.SqlServer.Server;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Web.Mvc;

namespace ProiectDaw.Models
{
    public class Comenzi
    {
        [Key, Column("Id_Comanda")]
        public int IdComanda { get; set; }

        public int Total { get; set; }
        //many to many
        public virtual ICollection<Bijuterii> Bijuterii { get; set; }
        // one to many
        [ForeignKey("User_id")]
        public int UserId { get; set; }
        public virtual User User { get; set; }
        [NotMapped]
        public IEnumerable<SelectListItem> UserList { get; set; }

    }
}