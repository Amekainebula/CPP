#include <bits/stdc++.h>
using namespace std;
int fa[1000006];

int finds(int x)
{
    return fa[x] == x ? x : fa[x] = finds(fa[x]);
}
void merge(int x, int y)
{
    int fx = finds(x), fy = finds(y);
    if (fx != fy)
    {
        fa[fx] = fy;
    }
}

void solve()
{
    int n;
    cin >> n;
    int ans = 1;
    for (int i = 0; i <= n + 1; i++)
    {
        fa[i] = i;
    }
    for (int i = 1; i <= n; i++)
    {
        int l, r;
        cin >> l >> r;
        if (ans)
        {
            r++;
            if (finds(l) == finds(r))
                ans = 0;
            else
                merge(l, r);
        }
    }
    cout << ans << "\n";
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
