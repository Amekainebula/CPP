#include <bits/stdc++.h>
using namespace std;

class sth
{
public:
    int w, v, c;
};

void solve()
{
    int n;
    cin >> n;
    vector<sth> a(n + 1);
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i].w >> a[i].v >> a[i].c;
    }
    auto cmp = [](sth a, sth b)
    {
        return a.c * b.w > a.w * b.c;
    };
    sort(a.begin() + 1, a.end(), cmp);
    int sum = a[n].v, ww = a[n].w;
    for (int i = n - 1; i >= 1; i--)
    {
        sum += a[i].v - ww * a[i].c;
        ww += a[i].w;
    }
    cout << sum << "\n";
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
