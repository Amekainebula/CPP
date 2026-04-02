#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl endl << flush
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int n, k;
vi a(55), b(55);
void q(int x, vi &vis)
{
    cout << "? " << x << endl;
    vis[x] = 1;
}
void Murasame()
{
    cin >> n >> k;
    vi vis(n + 1, 0);
    ff(i, 1, k)
    {
        q(i, vis);
        cin >> a[i];
    }
    ff(i, 1, k)
    {
        q(n - i + 1, vis);
        cin >> b[((n % k) - i + k) % k + 1];
    }
    bool ok = 0;
    vi temp;
    ff(i, 1, k)
    {
        if (a[i] != b[i])
        {
            ok = 1;
            temp.pb(i);
        }
    }
    if (!ok)
    {
        if (n == k * 2)
            cout << "! " << k << " " << k << endl;
        else
            cout << "! -1" << endl;
    }
    else
    {
        int l = k, r = n - k + 1;
        for (auto i : temp)
        {
            int mmax = (n - i) / k;
            int ll = max(1LL, l / k), rr = min(mmax, (r - 1) / k);
            int fr = -1;
            int m = 0;
            while (ll <= rr)
            {
                int mid = (ll + rr) / 2;
                int j = i + mid * k;
                if (j > n || vis[j])
                {
                    rr = mid - 1;
                    continue;
                }
                if (j <= l)
                {
                    ll = mid + 1;
                    continue;
                }
                q(j, vis);
                int tem;
                cin >> tem;
                if (tem == a[i])
                {
                    m = mid;
                    l = max(l, j);
                    ll = mid + 1;
                }
                else
                {
                    fr = j;
                    rr = mid - 1;
                }
            }
            if (fr != -1)
            {
                r = min(r, fr);
                if (l == r - 1)
                    break;
            }
            else if (m > 0)
            {
                l = max(l, i + m * k);
            }
        }
        if (l == r - 1)
            cout << "! " << l << " " << n - r + 1 << endl;
        else
            cout << "! -1" << endl;
    }
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
        Murasame();
    }
    return 0;
}