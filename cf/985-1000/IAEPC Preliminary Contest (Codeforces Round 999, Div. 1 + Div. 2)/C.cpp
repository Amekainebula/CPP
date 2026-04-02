#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
//#define mp make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define endl '\n'
using namespace std;
int a[200005],b[200005];
void solve()
{
    set<int> s1;
    set<int> s2;
    map<int, int> mp1;
    map<int, int> mp2;
    int n, m;
    ull sum1 = 0, sum2 = 0;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        sum1 += a[i];
        s1.insert(a[i]);
        mp1[a[i]]++;
    }
    for (int i = 1; i <= m; i++)
    {
        int x;
        cin >> x;
        b[i] = x;
        sum2 += x;
        s2.insert(x);
        mp2[x]++;
    }
    if (sum1 != sum2)
    {
        cout << "NO" << endl;
        return;
    }
    if (n == m)
    {
        for (int i = 1; i <= n; i++)
        {
            if (a[i] != b[i])
            {
                cout << "NO" << endl;
                return;
            }
        }
        cout << "YES" << endl;
        return;
    }
    vector<int> has, ned;
    set<int> hh, nn;
    map<int, int> hhh,nnn;
    for (auto m1 : mp1)
    {
        if (s2.find(m1.fi) != s2.end())
        {
            int temp = min(mp1[m1.fi], mp2[m1.fi]);
            mp1[m1.fi] -= temp;
            mp2[m1.fi] -= temp;
        }
        for(int i = 1; i <= m1.second; i++)
        {
            has.pb(m1.fi);
            hh.insert(m1.fi);
            hhh[m1.fi]++;
        }
    }
    for (auto m2 : mp2)
    {
        for(int i = 1; i <= m2.second; i++)
        {
            ned.pb(m2.fi);
            nn.insert(m2.fi);
            nnn[m2.fi]++;
        }
    }
    while (hh.size() && nn.size())
    {
        int flag = 0;
        vector<int> temp;
        for (int i = 0; i < has.size(); i += 2)
        {
            int x1 = has[i];
            int x2 = has[i + 1];
            if (fabs(x1 - x2) <= 1)
            {
                temp.insert(upper_bound(has.begin(), has.end(), x1 + x2), x1 + x2);
                hh.insert(x1 + x2);
                hhh[x1 + x2]++;
                hhh[x1]--;
                hhh[x2]--;
                if (hhh[x1] == 0)hh.erase(x1);
                if (hhh[x2] == 0)hh.erase(x2);
                flag = 1;
                if (nn.find(x1 + x2) != nn.end())
                {
                    nnn[x1 + x2]--;
                    hhh[x1 + x2]--;
                    if (hhh[x1 + x2] == 0)hh.erase(x1 + x2);
                    if (nnn[x1 + x2] == 0)nn.erase(x1 + x2);
                }
            }
            else
            { 
                temp.insert(upper_bound(has.begin(), has.end(), x1), x1);
                temp.insert(upper_bound(has.begin(), has.end(), x2), x2);
            }
        }
        if (has.size() % 2 == 1)
        {
            temp.insert(upper_bound(has.begin(), has.end(), has.back()), has.back());
        }
        has = temp;
    }
    if (hh.size() == 0 && nn.size() == 0)
    {
        cout << "YES" << endl;
    }
    else
    {
        cout << "NO" << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}