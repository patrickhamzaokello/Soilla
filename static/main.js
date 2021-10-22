const productCards = document.querySelectorAll(".product-card");
const childinputname = document.querySelector("#childnameinput");
const sponsorshipform = document.querySelector(".sponserdiv");
const displaySetting = sponsorshipform.style.display;


productCards.forEach((productCard) => {
  const productCardLink = productCard.querySelector(".product-card__link");

  const childNamegot = productCard.querySelector(".childname").innerHTML;

  // Make whole card clickable, but only if event target is NOT a specific card action inside <div class="product-card__actions">.
  productCard.addEventListener("click", (e) => {
    if (e.target.closest(".product-card__actions") === null) {
      // productCardLink.click();


      if (displaySetting == "block") {
        sponsorshipform.style.display = "none";
      } else {
        sponsorshipform.style.display = "grid";
      }

      console.log(childNamegot);
      childinputname.value = childNamegot;
    }
  });
});

function cancelsponsohip() {
    sponsorshipform.style.display = "none";
}
