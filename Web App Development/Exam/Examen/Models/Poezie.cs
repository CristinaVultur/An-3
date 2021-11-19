using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace Examen.Models
{
    public class Poezie
    {
        [Key]
        public int PoezieId { get; set; }
        [Required]
        public string Titlu { get; set; }

        [Required]
        public string Autor { get; set; }
        [Required]
        public int NumarStrofe { get; set; }
      
        
        public int VolumId { get; set; }
        [Required]
        public virtual Volum Volum { get; set; }

        [NotMapped]
        public IEnumerable<SelectListItem> VolumeList { get; set; }
    }
}