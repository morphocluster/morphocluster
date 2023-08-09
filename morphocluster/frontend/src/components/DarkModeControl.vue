<template>
    <div>
        <input
            v-model="dark_mode"
            id="dark-mode-checkbox"
            type="checkbox"
            class="switch-checkbox"
        />
        <label for="dark-mode-checkbox">Dark Mode</label>
    </div>
</template>

<script>
export default {
    name: "dark-mode-toggle",
    mounted() {
        // this.dark_mode = this.dark_mode;
    },
    computed: {
        dark_mode: {
            get() {
                const dark_mode = localStorage.getItem("dark-mode") == "true";
                console.log("dark-mode get", dark_mode);
                return dark_mode;
            },
            set(dark_mode) {
                console.log("dark-mode set", dark_mode);

                localStorage.setItem("dark-mode", dark_mode ? "true" : "false");

                if (dark_mode) {
                    document.documentElement.classList.add("dark-mode");
                } else {
                    document.documentElement.classList.remove("dark-mode");
                }
            },
        },
    },
};
</script>
