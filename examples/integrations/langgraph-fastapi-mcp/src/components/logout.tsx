"use client";
import { LogOut, LogOutIcon } from "lucide-react";
import { Button } from "./ui/button";
import { authClient } from "@/app/lib/auth-client";

export function Logout() {
  const handleLogout = async () => {
    await authClient.signOut();
  };

  return (
    <Button variant="outline" onClick={handleLogout}>
      Logout <LogOut className="size-4" />
    </Button>
  );
}
