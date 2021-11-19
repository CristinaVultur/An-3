using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.ComponentModel.DataAnnotations;

namespace ProiectDawAut.Models.MyValidation
{
    public class TenMultipleValidator : ValidationAttribute
    {
        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            var bijuterie = (Bijuterii)validationContext.ObjectInstance;
            int pret = bijuterie.Pret;
            bool cond = true;

            
            if (pret % 10 != 0)
            {
              cond = false;
            }
            
            return cond ? ValidationResult.Success : new ValidationResult("Nu e multiplu de 10!");
        }
    }
}