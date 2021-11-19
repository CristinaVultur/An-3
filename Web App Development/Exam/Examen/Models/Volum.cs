using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace Examen.Models
{
    public class Volum
    {
        [Key]
        public int VolumId { get; set; }
        [Required]
        public string Denumire { get; set; }
        // many-to-one relationship
        [Required]
        public virtual ICollection<Poezie> Poezii { get; set; }
    }
}