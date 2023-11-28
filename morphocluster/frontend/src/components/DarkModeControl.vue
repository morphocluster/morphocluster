<template>
    <v-switch v-model="DarkMode" label="Dark Mode"></v-switch>
</template>

<script>
export default {
    name: "dark-mode-toggle",
    mounted() {
        this.DarkMode = this._getDarkMode();
    },
    methods: {
        _getDarkMode() {
            const dark_mode = localStorage.getItem("dark-mode") == "true";
            console.log("dark-mode get", dark_mode);
            return dark_mode;
        }
    },
    computed: {
        DarkMode: {
            get() {
                return this._getDarkMode();
            },
            set(dark_mode) {
                console.log("dark-mode set", dark_mode);

                localStorage.setItem("dark-mode", dark_mode ? "true" : "false");

                this.$vuetify.theme.dark = dark_mode;
            },
        },
    },
};
</script>
